import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
np.set_printoptions(precision=3)

import os
import sys
import time
import copy
import json
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.stket import STKET

torch.set_num_threads(4)

"""------------------------------------some settings----------------------------------------"""
conf = Config()
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)

# Bulid Logger
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
file_path = os.path.join(conf.save_path, 'logs.log')
file_handler = logging.FileHandler(file_path)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('The CKPT saved here: {}'.format(conf.save_path))
logger.info('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    logger.info('{} : {}'.format(i, conf.args[i]))
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")

# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

trainPrior = json.load(open('data/TrainPrior.json', 'r'))

model = STKET(mode=conf.mode,
   attention_class_num=len(AG_dataset_train.attention_relationships),
   spatial_class_num=len(AG_dataset_train.spatial_relationships),
   contact_class_num=len(AG_dataset_train.contacting_relationships),
   obj_classes=AG_dataset_train.object_classes,
   N_layer_num=conf.N_layer,
   enc_layer_num=conf.enc_layer_num,
   dec_layer_num=conf.dec_layer_num,
   pred_contact_threshold=conf.pred_contact_threshold,
   window_size=conf.window_size,
   trainPrior=trainPrior,
   use_spatial_prior=conf.use_spatial_prior,
   use_temporal_prior=conf.use_temporal_prior).to(device=gpu_device)

if conf.model_path != 'None':
    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    logger.info('*'*50)
    logger.info('CKPT {} is loaded'.format(conf.model_path))

evaluator1 = [BasicSceneGraphEvaluator(mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='with') for _ in range(3)] 

evaluator2 = [BasicSceneGraphEvaluator(mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='no') for _ in range(3)]


# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    nll_loss = nn.NLLLoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()
    nll_loss = nn.NLLLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=0, factor=0.5, verbose=True, threshold=1e-3, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []

if conf.eval:

    model.eval()
    object_detector.is_train = False

    with torch.no_grad():
        for b, data in enumerate(tqdm(dataloader_test)):

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            pred = model(entry)

            if conf.enc_layer_num > 0:
                    evaluator1[0].evaluate_scene_graph(gt_annotation, pred, 'spatial')
                    evaluator2[0].evaluate_scene_graph(gt_annotation, pred, 'spatial') 

            if conf.dec_layer_num > 0:
                evaluator1[1].evaluate_scene_graph(gt_annotation, pred, 'temporal')
                evaluator2[1].evaluate_scene_graph(gt_annotation, pred, 'temporal')

            if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                evaluator1[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')
                evaluator2[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')

    logger.info('-------------------------Basic Metric-------------------------------')

    logger.info('-------------------------with constraint-------------------------------')    
    if conf.enc_layer_num > 0:
        evaluator1[0].print_stats(logger, 'spatial')
        evaluator1[0].reset_result()

    if conf.dec_layer_num > 0:
        evaluator1[1].print_stats(logger, 'temporal')
        evaluator1[1].reset_result()

    if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
        evaluator1[2].print_stats(logger, 'ensemble')
        evaluator1[2].reset_result()

    logger.info('-------------------------no constraint-------------------------------')    
    if conf.enc_layer_num > 0:
        evaluator2[0].print_stats(logger, 'spatial')
        evaluator2[0].reset_result()

    if conf.dec_layer_num > 0:
        evaluator2[1].print_stats(logger, 'temporal')
        evaluator2[1].reset_result()
        
    if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
        evaluator2[2].print_stats(logger, 'ensemble')
        evaluator2[2].reset_result()

else:

    for epoch in range(conf.nepoch):

        model.train()
        object_detector.is_train = True
        start = time.time()
        train_iter = iter(dataloader_train)
        test_iter = iter(dataloader_test)
        
        for b in tqdm(range(len(dataloader_train))):

            data = next(train_iter)
            
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]

            # Shape:
            # im_data: tensor, (num_frames, 3, height, weight)
            # im_info: tensor, (num_frames, 3)
            # gt_boxes: tensor, (num_frames, 1, 5)
            # num_boxes: tensor, (num_frames,)
            # gt_annotation: list(list), (num_frames,)

            # elements in gt_annotation[i]
            # [0] : dict, (person_bbox)
            # [1:] : dict, (class, bbox, attention_relationship, spatial_relationship, contacting_relationship, metadata, visible)

            # prevent gradients to FasterRCNN
            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

            # elements in entry:
            # boxes: tensor, (all_bbox_nums, 5)
            # labels: tensor, (all_bbox_nums,)
            # scores: tensor, (all_bbox_nums,)
            # im_idx: tensor, (relation_nums,)
            # pair_idx: tensor, (pair_nums, 2)
            # human_idx: tensor, (person_nums,)
            # features: tensor, (all_bbox_nums, 2048)
            # union_feat: tensor, (pair_num, 1024, 7, 7)
            # union_box: tensor, (pair_nums, 5)
            # spatial_masks:
            # attention_gt: list, (relation_nums,). e.g., [[0], [2], [3]]
            # spatial_gt: list, (relation_nums,). e.g., [[5], [3, 4], [3]]
            # contacting_gt: list, (relation_nums,). e.g., [[8], [10, 6], [8]]
            
            pred = model(entry)

            if conf.enc_layer_num > 0:
                spatial_attention_distribution, spatial_spatial_distribution, spatial_contact_distribution = pred["spatial_attention_distribution"], pred["spatial_spatial_distribution"], pred["spatial_contacting_distribution"]
            if conf.dec_layer_num > 0:
                temporal_attention_distribution, temporal_spatial_distribution, temporal_contact_distribution = pred["temporal_attention_distribution"], pred["temporal_spatial_distribution"], pred["temporal_contacting_distribution"]
            if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                ensemble_attention_distribution, ensemble_spatial_distribution, ensemble_contact_distribution = pred["ensemble_attention_distribution"], pred["ensemble_spatial_distribution"], pred["ensemble_contacting_distribution"]

            if conf.use_spatial_prior and conf.spatial_prior_loss:
                spatial_prior_attention_distribution, spatial_prior_spatial_distribution, spatial_prior_contact_distribution = pred["spatial_prior_attention_distribution"], pred["spatial_prior_spatial_distribution"], pred["spatial_prior_contacting_distribution"]
            if conf.use_temporal_prior and conf.temporal_prior_loss:
                temporal_prior_attention_distribution, temporal_prior_spatial_distribution, temporal_prior_contact_distribution = pred["temporal_prior_attention_distribution"], pred["temporal_prior_spatial_distribution"], pred["temporal_prior_contacting_distribution"]

            attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=im_data.device).squeeze()
            if not conf.bce_loss:
                # multi-label margin loss or adaptive loss
                spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=im_data.device)
                contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=im_data.device)
                for i in range(len(pred["spatial_gt"])):
                    spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                    contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

            else:
                # bce loss
                spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=im_data.device)
                contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=im_data.device)
                for i in range(len(pred["spatial_gt"])):
                    spatial_label[i, pred["spatial_gt"][i]] = 1
                    contact_label[i, pred["contacting_gt"][i]] = 1

            losses = {}
            if conf.mode == 'sgcls' or conf.mode == 'sgdet':
                losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

            if conf.enc_layer_num > 0:
                losses["spatial_attention_relation_loss"] = ce_loss(spatial_attention_distribution, attention_label)
            if conf.dec_layer_num > 0:
                losses["temporal_attention_relation_loss"] = ce_loss(temporal_attention_distribution, attention_label)
            if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                losses["ensemble_attention_relation_loss"] = ce_loss(ensemble_attention_distribution, attention_label)

            if conf.use_spatial_prior and conf.spatial_prior_loss: 
                losses["spatial_prior_attention_relation_loss"] = ce_loss(spatial_prior_attention_distribution, attention_label)
            if conf.use_temporal_prior and conf.temporal_prior_loss:
                losses["temporal_prior_attention_relation_loss"] = ce_loss(temporal_prior_attention_distribution, attention_label)

            if not conf.bce_loss:
                if conf.enc_layer_num > 0:
                    losses["spatial_spatial_relation_loss"] = mlm_loss(spatial_spatial_distribution, spatial_label)
                    losses["spatial_contact_relation_loss"] = mlm_loss(spatial_contact_distribution, contact_label)
                if conf.dec_layer_num > 0:
                    losses["temporal_spatial_relation_loss"] = mlm_loss(temporal_spatial_distribution, spatial_label)
                    losses["temporal_contact_relation_loss"] = mlm_loss(temporal_contact_distribution, contact_label)
                if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                    losses["ensemble_spatial_relation_loss"] = mlm_loss(ensemble_spatial_distribution, spatial_label)
                    losses["ensemble_contact_relation_loss"] = mlm_loss(ensemble_contact_distribution, contact_label)

                if conf.use_spatial_prior and conf.spatial_prior_loss:
                    losses["spatial_prior_spatial_relation_loss"] = mlm_loss(spatial_prior_spatial_distribution, spatial_label)
                    losses["spatial_prior_contact_relation_loss"] = mlm_loss(spatial_prior_contact_distribution, contact_label)

                if conf.use_temporal_prior and conf.temporal_prior_loss:
                    losses["temporal_prior_spatial_relation_loss"] = mlm_loss(temporal_prior_spatial_distribution, spatial_label)
                    losses["temporal_prior_contact_relation_loss"] = mlm_loss(temporal_prior_contact_distribution, contact_label)
            else:
                if conf.enc_layer_num > 0:
                    losses["spatial_spatial_relation_loss"] = bce_loss(spatial_spatial_distribution, spatial_label)
                    losses["spatial_contact_relation_loss"] = bce_loss(spatial_contact_distribution, contact_label)
                if conf.dec_layer_num > 0:
                    losses["temporal_spatial_relation_loss"] = bce_loss(temporal_spatial_distribution, spatial_label)
                    losses["temporal_contact_relation_loss"] = bce_loss(temporal_contact_distribution, contact_label)
                if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                    losses["ensemble_spatial_relation_loss"] = bce_loss(ensemble_spatial_distribution, spatial_label)
                    losses["ensemble_contact_relation_loss"] = bce_loss(ensemble_contact_distribution, contact_label)

                if conf.use_spatial_prior and conf.spatial_prior_loss:
                    losses["spatial_prior_spatial_relation_loss"] = bce_loss(spatial_prior_spatial_distribution, spatial_label)
                    losses["spatial_prior_contact_relation_loss"] = bce_loss(spatial_prior_contact_distribution, contact_label)

                if conf.use_temporal_prior and conf.temporal_prior_loss:
                    losses["temporal_prior_spatial_relation_loss"] = bce_loss(temporal_prior_spatial_distribution, spatial_label)
                    losses["temporal_prior_contact_relation_loss"] = bce_loss(temporal_prior_contact_distribution, contact_label)

            optimizer.zero_grad()
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

            print_freq = 1000
            if b % print_freq == 0:
                time_per_batch = (time.time() - start) / print_freq
                logger.info("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(
                            epoch, b, len(dataloader_train),
                            time_per_batch, len(dataloader_train) * time_per_batch / 60))

                mn = pd.concat(tr[-print_freq:], axis=1).mean(1)
                logger.info(mn)
                start = time.time()

        torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
        logger.info("*" * 40)
        logger.info("save the checkpoint after {} epochs".format(epoch))
        
        model.eval()
        object_detector.is_train = False
        with torch.no_grad():
            for b in tqdm(range(len(dataloader_test))):
                data = next(test_iter)

                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
                gt_annotation = AG_dataset_test.gt_annotations[data[4]]

                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                pred = model(entry)

                if conf.enc_layer_num > 0:
                    evaluator1[0].evaluate_scene_graph(gt_annotation, pred, 'spatial')
                    evaluator2[0].evaluate_scene_graph(gt_annotation, pred, 'spatial') 

                if conf.dec_layer_num > 0:
                    evaluator1[1].evaluate_scene_graph(gt_annotation, pred, 'temporal')
                    evaluator2[1].evaluate_scene_graph(gt_annotation, pred, 'temporal')

                if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                    evaluator1[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')
                    evaluator2[2].evaluate_scene_graph(gt_annotation, pred, 'ensemble')

            logger.info('-----------') 

        score = np.mean(evaluator1[1].result_dict[conf.mode + "_recall"][20])
        scheduler.step(score)

        logger.info('-------------------------Basic Metric-------------------------------')

        logger.info('-------------------------with constraint-------------------------------')    
        if conf.enc_layer_num > 0:
            evaluator1[0].print_stats(logger, 'spatial')
            evaluator1[0].reset_result()

        if conf.dec_layer_num > 0:
            evaluator1[1].print_stats(logger, 'temporal')
            evaluator1[1].reset_result()

        if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
            evaluator1[2].print_stats(logger, 'ensemble')
            evaluator1[2].reset_result()

        logger.info('-------------------------no constraint-------------------------------')    
        if conf.enc_layer_num > 0:
            evaluator2[0].print_stats(logger, 'spatial')
            evaluator2[0].reset_result()

        if conf.dec_layer_num > 0:
            evaluator2[1].print_stats(logger, 'temporal')
            evaluator2[1].reset_result()
            
        if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
            evaluator2[2].print_stats(logger, 'ensemble')
            evaluator2[2].reset_result()

