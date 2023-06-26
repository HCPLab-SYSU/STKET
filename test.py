import numpy as np
np.set_printoptions(precision=4)

import json
import copy
import torch

from tqdm import tqdm
from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.new_evaluation_recall import DynamicSceneGraphEvaluator
from lib.object_detector import detector
from lib.stket import STKET

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

trainPrior = json.load(open('data/TrainPrior.json', 'r'))

model = STKET(mode=conf.mode,
   attention_class_num=len(AG_dataset.attention_relationships),
   spatial_class_num=len(AG_dataset.spatial_relationships),
   contact_class_num=len(AG_dataset.contacting_relationships),
   obj_classes=AG_dataset.object_classes,
   enc_layer_num=conf.enc_layer_num,
   dec_layer_num=conf.dec_layer_num,
   pred_contact_threshold=conf.pred_contact_threshold,
   window_size=conf.window_size,
   trainPrior=trainPrior,
   use_spatial_prior=conf.use_spatial_prior,
   use_temporal_prior=conf.use_temporal_prior).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))

evaluator1 = [BasicSceneGraphEvaluator(mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with') for _ in range(3)] 

evaluator2 = [BasicSceneGraphEvaluator(mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no') for _ in range(3)]

evaluator3 = [DynamicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint=True) for _ in range(3)]

evaluator4 = [DynamicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint=False) for _ in range(3)]

with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        pred = model(entry)

        evaluator1[0].evaluate_scene_graph(gt_annotation, pred, 'spatial'), evaluator1[1].evaluate_scene_graph(gt_annotation, pred, 'temporal'), evaluator1[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')
        evaluator2[0].evaluate_scene_graph(gt_annotation, pred, 'spatial'), evaluator2[1].evaluate_scene_graph(gt_annotation, pred, 'temporal'), evaluator2[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')
        evaluator3[0].evaluate_scene_graph(gt_annotation, pred, 'spatial'), evaluator3[1].evaluate_scene_graph(gt_annotation, pred, 'temporal'), evaluator3[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')
        evaluator4[0].evaluate_scene_graph(gt_annotation, pred, 'spatial'), evaluator4[1].evaluate_scene_graph(gt_annotation, pred, 'temporal'), evaluator4[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')

print('-------------------------Basic Metric-------------------------------')

print('-------------------------with constraint-------------------------------')    
evaluator1[0].print_stats(None, 'spatial'), evaluator1[1].print_stats(None, 'temporal'), evaluator1[2].print_stats(None, 'ensemble')

print('-------------------------no constraint-------------------------------')    
evaluator2[0].print_stats(None, 'spatial'), evaluator2[1].print_stats(None, 'temporal'), evaluator2[2].print_stats(None, 'ensemble')
print('-------------------------Dynamic Metric-----------------------------')

print('-------------------------with constraint----------------------------')
evaluator3[0].print_stats(None, 'spatial'), evaluator3[1].print_stats(None, 'temporal'), evaluator3[2].print_stats(None, 'ensemble')

print('-------------------------no constraint------------------------------')
evaluator4[0].print_stats(None, 'spatial'), evaluator4[1].print_stats(None, 'temporal'), evaluator4[2].print_stats(None, 'ensemble')
