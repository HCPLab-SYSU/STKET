import os
import copy
import json
import pandas as pd
from tqdm import tqdm

import numpy as np
np.set_printoptions(precision=4)

from lib.config import Config
from lib.object_detector import detector
from dataloader.action_genome import AG, cuda_collate_fn

import torch

"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
     print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=False, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)

AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")

# freeze the detection backbone
object_detector = detector(train=False, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

def countAnno(AG_dataset, dataloader, object_detector, annoPath):

    def getRelArray(a_rel, s_rel, c_rel, a_rel_num, s_rel_num, c_rel_num):

        a_rel_res, s_rel_res, c_rel_res = np.zeros(a_rel_num, dtype=np.int), np.zeros(s_rel_num, dtype=np.int), np.zeros(c_rel_num, dtype=np.int)

        a_rel, s_rel, c_rel = n['attention_relationship'].numpy().tolist(), n['spatial_relationship'].numpy().tolist(), n['contacting_relationship'].numpy().tolist()

        a_rel_res[a_rel] = 1
        s_rel_res[s_rel] = 1
        c_rel_res[c_rel] = 1

        result = np.concatenate((a_rel_res, s_rel_res, c_rel_res))
        result = np.expand_dims(result, axis=0)

        return result

    print('object classes num: {}, total relationships num: {}, a_rel num: {}, s_rel num: {}, c_rel num {}.'.format(
          len(AG_dataset.object_classes), len(AG_dataset.relationship_classes), 
          len(AG_dataset.attention_relationships), len(AG_dataset.spatial_relationships), len(AG_dataset.contacting_relationships)
          ))

    a_rel_num, s_rel_num, c_rel_num = len(AG_dataset.attention_relationships), len(AG_dataset.spatial_relationships), len(AG_dataset.contacting_relationships)

    annos = {}

    for b, data in enumerate(tqdm(dataloader)):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt = AG_dataset.gt_annotations[data[4]]

        _annos = {}

        for idx, frame_gt in enumerate(gt):

            for m, n in enumerate(frame_gt[1:]):

                if n['class'] not in _annos:
                    _annos[n['class']] = []

                _annos[n['class']].append(getRelArray(n['attention_relationship'], n['spatial_relationship'], n['contacting_relationship'], a_rel_num, s_rel_num, c_rel_num))

        for obj_class_idx, anno in _annos.items():
            rel_anno = np.concatenate(anno, axis=0) # ndarray, (current_frame_num, total_rel_num)

            if obj_class_idx not in annos:
                annos[obj_class_idx] = {'rel_anno': [], 'max_frame_num': 0}
            
            annos[obj_class_idx]['rel_anno'].append(rel_anno)
            annos[obj_class_idx]['max_frame_num'] = max(annos[obj_class_idx]['max_frame_num'], rel_anno.shape[0])

    total_rel_num = a_rel_num + s_rel_num + c_rel_num
    for obj_class_idx, anno in annos.items():

        result = np.zeros((len(anno['rel_anno']), anno['max_frame_num'], total_rel_num), dtype=np.int)

        for idx in range(len(anno['rel_anno'])):

            current_frame_num = anno['rel_anno'][idx].shape[0]
            result[idx, :current_frame_num] = anno['rel_anno'][idx]
            result[idx, current_frame_num:] = -1

        result = np.reshape(result, (-1, total_rel_num))

        np.savetxt(os.path.join(annoPath, 'objIndex_{}_videoNum_{}_frameNum_{}.txt'.format(obj_class_idx, len(anno['rel_anno']), anno['max_frame_num'])), 
                   result,
                   fmt='%d',
                   delimiter=',')

def countPrior(AG_dataset, dataloader, object_detector, mode='Train'):

    annoPath = 'data/TrainAnnos' if mode == 'Train' else 'data/TestAnnos'
    if not os.path.exists(annoPath):
        os.mkdir(annoPath)
        countAnno(AG_dataset, dataloader, object_detector, annoPath)

    a_rel_num, s_rel_num, c_rel_num = len(AG_dataset.attention_relationships), len(AG_dataset.spatial_relationships), len(AG_dataset.contacting_relationships)
    total_rel_num = a_rel_num + s_rel_num + c_rel_num

    prior = {}
  
    for annoName in os.listdir(annoPath):

        obj_cls_index, video_num, max_frame_num = int(annoName.split('_')[1]), int(annoName.split('_')[3]), int(annoName.split('_')[5][:-4])
        
        prior[obj_cls_index] = {'obj_cls_name': AG_dataset.object_classes[obj_cls_index],
                                'c_rel_name': [],
                                'c_rel_idx': [],
                                'initial_prior': [],
                                'specific_prior': []}

        anno = np.loadtxt(os.path.join(annoPath, annoName), dtype=np.int, delimiter=',') # numpy.ndarray, (video_num * max_frame_num, total_rel_num)
        reshape_anno = np.reshape(anno, (video_num, max_frame_num, total_rel_num))       # numpy.ndarray, (video_num, max_frame_num, total_rel_num)
        c_rel_anno = anno[:, -c_rel_num:]                                                # numpy.ndarray, (video_num * max_frame_num, c_rel_num)     

        # get contains contacting relationship index and name
        prior[obj_cls_index]['c_rel_idx'] = np.where(np.sum(c_rel_anno == 1, axis=0))[0].tolist()
        prior[obj_cls_index]['c_rel_name'] = [AG_dataset.contacting_relationships[idx] for idx in prior[obj_cls_index]['c_rel_idx']]

        # get initial prior
        prior[obj_cls_index]['initial_prior'] = anno[anno[:, 0] != -1 ].mean(axis=0).tolist()

        # get specific prior
        for c_rel_idx in prior[obj_cls_index]['c_rel_idx']:

            _prior = []

            for video_idx in range(video_num):

                frame_num = max_frame_num-1 if reshape_anno[video_idx, -1, 0] != -1 else np.where(reshape_anno[video_idx, :, 0] != -1)[0].tolist()[-1]
                _anno = reshape_anno[video_idx, :frame_num]

                _idx = np.where(_anno[:, a_rel_num + s_rel_num + c_rel_idx] == 1)[0] + 1

                if reshape_anno[video_idx][_idx].shape[0] != 0:
                    _prior.append(np.expand_dims(reshape_anno[video_idx][_idx].mean(axis=0), axis=0))

            _prior = np.concatenate(_prior, axis=0) if len(_prior) > 1 else (_prior[0] if len(_prior) == 1 else np.zeros((1, total_rel_num)))
            _prior = _prior.mean(axis=0)

            prior[obj_cls_index]['specific_prior'].append(_prior.tolist())

    jsonPath = 'data/TrainPrior.json' if mode == 'Train' else 'data/TestPrior.json'
    with open(jsonPath, 'w') as f:
        f.write(json.dumps(prior))

            
if __name__=='__main__':
    # train set
    print("--------------------Train Set-------------------------")
    countPrior(AG_dataset_train, dataloader_train, object_detector, mode='Train')
    print("------------------------------------------------------")

    # test set
    print("--------------------Test Set--------------------------")
    countPrior(AG_dataset_test, dataloader_test, object_detector, mode='Test')
    print("------------------------------------------------------")
