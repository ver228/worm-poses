#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

from pathlib import Path 
from argparse import ArgumentParser

from .engines import train_poses
from .data import SkelMapsFlow, SkelMapsFlowValidation, get_outputs_sizes
from .models import PoseDetector, get_keypointrcnn
from .utils import get_device
from .configs import read_flow_config, read_model_config

import yaml
import datetime
import torch
import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
mp.set_start_method('fork', force=True)

def save_metadata(opt : ArgumentParser, save_dir : Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    fname = save_dir / 'config.yaml'
    if fname.exists():
        raise FileExistsError('There exist a `{fname}`, it is likely you are trying to overwrite a previous experiment.')
    with open(fname, 'w') as fid:
        yaml.safe_dump(opt.__dict__, fid)

def get_model_from_args(conf_args):
    n_segments_out, n_affinity_maps_out = get_outputs_sizes(n_segments=conf_args['n_segments'], 
                                                            PAF_seg_dist = conf_args['PAF_seg_dist'], 
                                                            fold_skeleton = conf_args['fold_skeleton']
                                       )
    if 'backbone' in conf_args:
        return get_keypointrcnn(backbone = conf_args['backbone'],
                                 num_classes = 2, 
                                 num_keypoints = n_segments_out
        )
    else:
        return PoseDetector(
                n_segments = n_segments_out, 
                n_affinity_maps = n_affinity_maps_out, 
                n_stages = conf_args['n_stages'],
                features_type = conf_args['features_type'],
                use_head_loss = conf_args['use_head_loss'],
                pose_loss_type = conf_args['pose_loss_type']
                )

    

def train(opt):
    flow_args = read_flow_config(opt.flow_config)
    conf_args = read_model_config(opt.model_config)
    
    data_dir = Path(opt.data_dir)
    
    log_dir = Path(opt.save_dir) / opt.flow_config
    
    return_bboxes = False
    return_half_bboxes = False
    if not 'openpose' in opt.model_config:
        if 'halfboxes' in opt.flow_config:
            return_half_bboxes = True
        else:
            return_bboxes = True
            
    
    train_flow = SkelMapsFlow(root_dir = data_dir, 
                             set2read =  'train', 
                             #set2read = 'validation',
                             samples_per_epoch = opt.samples_per_epoch,
                             return_key_value_pairs = True,
                             PAF_seg_dist = conf_args['PAF_seg_dist'],
                             n_segments = conf_args['n_segments'],
                             fold_skeleton = conf_args['fold_skeleton'],
                             return_bboxes = return_bboxes,
                             return_half_bboxes = return_half_bboxes,
                             **flow_args
                             )

    val_flow = SkelMapsFlowValidation(root_dir = data_dir, 
                             set2read = 'validation',
                             return_key_value_pairs = True,
                             PAF_seg_dist = conf_args['PAF_seg_dist'],
                             n_segments = conf_args['n_segments'],
                             fold_skeleton = conf_args['fold_skeleton'],
                             return_bboxes = return_bboxes,
                             return_half_bboxes = return_half_bboxes,
                             **flow_args
                             )
    model = get_model_from_args(conf_args)
    if opt.init_model_path:
        state = torch.load(opt.init_model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
    
    device = get_device(opt.cuda_id)
    lr_scheduler = None
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = opt.lr, weight_decay=opt.weight_decay)
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    
    basename = f'{opt.flow_config}_{opt.model_config}_{opt.loss_type}_{date_str}'
    save_metadata(opt, log_dir / basename)
    train_poses(basename,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        lr_scheduler = lr_scheduler,
        
        batch_size = opt.batch_size,
        n_epochs = opt.n_epochs,
        num_workers = opt.num_workers,
        save_frequency = opt.save_frequency
        )


parser = ArgumentParser()
parser.add_argument('--flow_config', type=str, default='v5', help='Flow (Augmentations) config file to be used.')
parser.add_argument('--model_config', type=str, default='openpose', help='Model config to be used')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU number to be used')
parser.add_argument('--save_dir', type=str, help='Directory where the model is going to be saved')
parser.add_argument('--data_dir', type=str, help='Directory where the source data is stored')
parser.add_argument('--batch_size', type=int, default=16, help ='Number of images used per training step')
parser.add_argument('--loss_type', type=str, default='maxlikelihood', help="NOTE: Leave this parameter as it is for the moment")
parser.add_argument('--samples_per_epoch', type=int, default=12288, help="Number of images generated done per epoch")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default=1, help="Number of epochs")
parser.add_argument('--num_workers', type=int, default=4, help="Number of workers used for the data loader.")
parser.add_argument('--save_frequency', type=int, default=200, help="Number of epochs where the model is going to be saved.")
parser.add_argument('--init_model_path', type=str, default='', help="Initial path of a model pretrained model")

if __name__ == '__main__':
    opt = parser.parse_args()
    train(opt)