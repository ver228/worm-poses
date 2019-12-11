#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:10:33 2019

@author: avelinojaver
"""
import torch
from pathlib import Path
import shutil

def get_device(cuda_id):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device


def get_optimizer(optimizer_name, model, lr, weight_decay = 0.0, momentum = 0.9, weigth_decay_no_bias = False):
    
    if weigth_decay_no_bias:
       model_params =  _weight_decay_no_bias(model, weight_decay)
    else:
        model_params = [{'params' : [x for x in  model.parameters() if x.requires_grad], 'weight_decay' : weight_decay}]
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr = lr, momentum = momentum)
    else:
        raise ValueError('Invalid optimizer name {}'.format(optimizer_name))
    return optimizer


def get_scheduler(lr_scheduler_name, optimizer):
    if not lr_scheduler_name:
        lr_scheduler = None
    elif lr_scheduler_name.startswith('stepLR'):
        #'stepLR-3-0.1'
        _, step_size, gamma = lr_scheduler_name.split('-')
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size = int(step_size),
                                                       gamma = float(gamma)
                                                       )
    elif lr_scheduler_name.startswith('cosineLR'):
        #'cosineLR-200'
        _, T_max = lr_scheduler_name.split('-')
        lr_scheduler = lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(T_max))
    else:
        raise ValueError(f'Not implemented {lr_scheduler_name}')
    return lr_scheduler

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / filename
    
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = save_dir / 'model_best.pth.tar'
        shutil.copyfile(checkpoint_path, best_path)