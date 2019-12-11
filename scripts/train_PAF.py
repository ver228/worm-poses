#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.trainer import train_poses
from worm_poses.flow import SkelMapsFlow, SkelMapsFlowValidation
from worm_poses.models import PoseDetector, get_keypointrcnn
from worm_poses.utils import get_device

import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
mp.set_start_method('fork', force=True)

import datetime
import torch


log_dir_root_dflt = Path.home() / 'workspace/WormData/worm-poses/results/'

root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
#root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'

flow_args = dict(
            data_types = ['from_tierpsy', 'manual'],
             negative_src = 'from_tierpsy_negative.p.zip',
             scale_int = (0, 255),
             
             roi_size = 256,
                 crop_size_lims = (50, 180),
                 negative_size_lims = (5, 180),
                 n_rois_lims = (1, 3),
                 int_expansion_range = (0.7, 1.3),
                 int_offset_range = (-0.2, 0.2),
                 blank_patch_range = (1/8, 1/3),
                 zoom_range = None,
                 
                 
            )

    #%%


available_models = {
    'openpose'  : dict(
            PAF_seg_dist = 5,
            n_segments = 49,
            n_stages = 6,
            features_type = 'vgg19'
        ),
    'openpose+light'  : dict(
            PAF_seg_dist = 2,
            n_segments = 17,
            n_stages = 4,
            features_type = 'vgg11'
        ),
    
    'keypointrcnn+resnet50' : dict(
            backbone = 'resnet50',
            PAF_seg_dist = None,
            n_segments = 49,
            
        ),
    'keypointrcnn+resnet18'  : dict(
            backbone = 'resnet18',
            PAF_seg_dist = None,
            n_segments = 49,
            
        ),
    
    }

def train_PAF(
            data_type = 'v2',
            model_name = 'openpose',
            cuda_id = 0,
            log_dir_root = log_dir_root_dflt,
            batch_size = 16,
            num_workers = 1,
            roi_size = 96,
            loss_type = 'maxlikelihood',
            lr = 1e-4,
            weight_decay = 0.0,
            n_epochs = 1, #1000,
            save_frequency = 200
            ):
    
    
    log_dir = log_dir_root / data_type
    
    return_bboxes = False if 'openpose' in model_name else True
    
    model_args = available_models[model_name]
    train_flow = SkelMapsFlow(root_dir = root_dir, 
                             set2read =  'train', 
                             #set2read = 'validation',
                             #samples_per_epoch = 1000,
                             return_key_value_pairs = True,
                             PAF_seg_dist = model_args['PAF_seg_dist'],
                             n_segments = model_args['n_segments'],
                             return_bboxes = return_bboxes,
                             **flow_args
                             )

    val_flow = SkelMapsFlowValidation(root_dir = root_dir, 
                             set2read = 'validation',
                             return_key_value_pairs = True,
                             PAF_seg_dist = model_args['PAF_seg_dist'],
                             n_segments = model_args['n_segments'],
                             return_bboxes = return_bboxes,
                             **flow_args
                             )
    
    
    if 'openpose' in model_name:
        model = PoseDetector(
                n_segments = train_flow.n_segments_out, 
                n_affinity_maps = train_flow.n_affinity_maps_out, 
                n_stages = model_args['n_stages'],
                features_type = model_args['features_type']
                )
    else:
        model = get_keypointrcnn(backbone = model_args['backbone'],
                                 num_classes = 2, 
                                 num_keypoints = train_flow.n_segments_out
                                 )
        
    
    device = get_device(cuda_id)
    lr_scheduler = None
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    
    basename = f'{data_type}_{model_name}_{loss_type}_{date_str}_adam_lr{lr}_wd{weight_decay}_batch{batch_size}'
        
    train_poses(basename,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        lr_scheduler = lr_scheduler,
        
        batch_size = batch_size,
        n_epochs = n_epochs,
        num_workers = num_workers,
        init_model_path = None,
        save_frequency = save_frequency
        )


if __name__ == '__main__':
    import fire
    
    fire.Fire(train_PAF)
    