#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
import os
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.trainer import trainer_detector, get_device, log_dir_root_dflt
from worm_poses.flow import RandomFlowDetection


pretrained_path = Path.home() / 'workspace/pytorch/pretrained_models/'
if pretrained_path.exists():
    os.environ['TORCH_HOME'] = str(pretrained_path)
from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn, keypointrcnn_resnet18_fpn


from available_datasets import data_types_dflts

import datetime
import torch

def train_detector(
            data_type = 'detection-clusters',
            model_name = 'fasterrcnn',
            cuda_id = 0,
            log_dir_root = log_dir_root_dflt,
            batch_size = 16,
            num_workers = 1,
            roi_size = 512,
            lr = 1e-4,
            weight_decay = 0.0,
            is_fixed_width = False,
            **argkws
            ):
    
    
    log_dir = log_dir_root / data_type
    
    dflts = data_types_dflts[data_type]
    root_dir =  dflts['root_dir']
    flow_args =  dflts['flow_args']
    
    
    flow_train = RandomFlowDetection(root_dir = root_dir, 
                                    set2read = 'train', #
                                    roi_size = roi_size,
                                    epoch_size = 5760,
                                    **flow_args
                                    )
    
    if 'keypointrcnn' in model_name:
        
        if model_name == 'keypointrcnn+resnet18':
            _build_func = keypointrcnn_resnet18_fpn
        else:
            _build_func = keypointrcnn_resnet50_fpn
       
        model = _build_func(
                            num_classes = 2, 
                            num_keypoints = 25,
                            min_size = roi_size,
                            max_size = roi_size,
                            image_mean = [0, 0, 0],
                            image_std = [1., 1., 1.]
                            )
    elif model_name == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(
                                    num_classes = 2, 
                                    min_size = roi_size,
                                    max_size = roi_size,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.]
                                    )
    
    else:
        raise ValueError(f'Not implemented {model_name}')
    
    
    device = get_device(cuda_id)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    
    
    basename = f'{data_type}_{model_name}_{date_str}_adam_lr{lr}_wd{weight_decay}_batch{batch_size}'
        
    
    trainer_detector(basename,
        model,
        device,
        flow_train,
        optimizer,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )


if __name__ == '__main__':
    import fire
    
    fire.Fire(train_detector)
    