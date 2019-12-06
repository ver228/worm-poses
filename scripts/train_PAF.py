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

from worm_poses.trainer import train_skeleton_maps, get_device, log_dir_root_dflt
from worm_poses.flow import SkelMapsRandomFlow, SkelMapsSimpleFlow, read_data_files
from worm_poses.models import OpenPoseCPM, OpenPoseCPMLoss, CPM_PAF,  CPM_Loss, PretrainedBackBone, get_preeval_func #CPM_PAF_Loss,
from available_datasets import data_types_dflts

import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
mp.set_start_method('fork', force=True)

import datetime
import torch


train_data = None
val_data = None


def train_PAF(
            data_type = 'v1',
            model_name = 'PAF+CPM',
            loss_type = 'mse',
            cuda_id = 0,
            log_dir_root = log_dir_root_dflt,
            batch_size = 16,
            num_workers = 1,
            roi_size = 96,
            lr = 1e-4,
            weight_decay = 0.0,
            is_fixed_width = False,
            **argkws
            ):
    
    
    log_dir = log_dir_root / data_type
    
    dflts = data_types_dflts[data_type]
    root_dir =  dflts['root_dir']
    flow_args =  dflts['flow_args']
    
    
    
    flow_args['width2sigma'] = -1 if loss_type == 'maxlikelihood' else flow_args['width2sigma']
    is_PAF = ('PAF' in model_name) or (model_name == 'openpose')
    
    
    train_data = read_data_files(root_dir = root_dir, set2read = 'train')
    val_data = read_data_files(root_dir = root_dir, set2read = 'validation')
    flow_train = SkelMapsRandomFlow(data = train_data,
                                    roi_size = roi_size,
                                    epoch_size = 23040,
                                    return_affinity_maps = is_PAF,
                                    is_fixed_width = is_fixed_width,
                                    **flow_args
                                    )
    
    
    flow_val = SkelMapsSimpleFlow(data = val_data,
                                    roi_size = roi_size,
                                    return_raw_skels = True,
                                    width2sigma = flow_args['width2sigma'],
                                    return_affinity_maps = is_PAF,
                                    is_fixed_width = is_fixed_width
                                    )
    
    if 'vgg19' in model_name:
        backbone = PretrainedBackBone('vgg19', pretrained = False)
    elif 'resnet50' in model_name:
        backbone = PretrainedBackBone('resnet50', pretrained = False)
    else:
        backbone = None
    
    
    if 'CPM' in model_name:
        model = CPM_PAF(n_segments = flow_train.n_skel_maps_out, 
                         n_affinity_maps = flow_train.n_affinity_maps_out, 
                         same_output_size = True, 
                         backbone = backbone,
                         is_PAF = is_PAF
                         )
        
    elif model_name == 'openpose':
        model = OpenPoseCPM(
                n_segments = flow_train.n_skel_maps_out, 
                n_affinity_maps = flow_train.n_affinity_maps_out, 
                
                )
    else:
        raise ValueError(f'Not implemented {model_name}')
    
    
    if model_name == 'openpose':
        criterion_func = OpenPoseCPMLoss
    elif is_PAF:
        criterion_func = CPM_PAF_Loss
    else:
        criterion_func = CPM_Loss
    
    
    if loss_type == 'mse':
        criterion = criterion_func()
    elif loss_type == 'maxlikelihood':
        criterion = criterion_func(is_maxlikelihood = True)
    else:
        raise ValueError(f'Not implemented {model_name}')
    
    preeval_func = get_preeval_func(loss_type)
    
    device = get_device(cuda_id)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    
    str_is_fixed = '-fixW' if is_fixed_width else ''
    basename = f'{data_type}{str_is_fixed}_{model_name}_{loss_type}_{date_str}_adam_lr{lr}_wd{weight_decay}_batch{batch_size}'
        
    train_skeleton_maps(basename,
        model,
        device,
        flow_train,
        flow_val,
        criterion,
        optimizer,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        preeval_func = preeval_func,
        **argkws
        )


if __name__ == '__main__':
    import fire
    
    fire.Fire(train_PAF)
    