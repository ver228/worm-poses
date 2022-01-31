#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
#import multiprocessing as mp
#mp.set_start_method('spawn', force=True)

from ..utils import save_checkpoint
from ..data import collate_to_dict

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import tqdm

__all__ = ['train_poses']

def _target_to_device(target_ori, device):
    target = {}
    for k,v in target_ori.items():
        if isinstance(v, torch.Tensor):
            v_new = v.to(device)
        elif isinstance(v, (list, tuple)):
            v_new = [x.to(device) for x in v]
        else:
            raise ValueError(f'type `{type(v)}` not valid')
        target[k] = v_new
    return target


def train_one_epoch(basename, model, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    train_avg_losses = defaultdict(int)
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, targets in pbar:
        
        images = images.to(device)
        targets = [_target_to_device(target, device) for target in targets]
        
        losses = model(images, targets)
        
        loss = sum([x for x in losses.values()])
        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(model.parameters(), 0.5) # I was having problems here before. I am not completely sure this makes a difference now
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        for k,l in losses.items():
            train_avg_losses[k] += l.item()
     
        
    train_avg_losses = {k: loss / len(data_loader) for k, loss in train_avg_losses.items()} #average loss
    train_avg_loss = sum([x for x in train_avg_losses.values()]) # total loss
    
    #save data into the logger
    for k, loss in train_avg_losses.items():
        logger.add_scalar('train_' + k, loss, epoch)
    logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
    
    return train_avg_loss

@torch.no_grad()
def evaluate_one_epoch(basename, model, data_loader, device, epoch, logger):
    model.eval()
    
    header = f'{basename} Test Epoch: [{epoch}]'
    
    model_time_avg = 0
    test_avg_losses = defaultdict(int)
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for ii, (images, targets) in enumerate(pbar):
        model_time = time.time()
        
        images = images.to(device)
        targets = [_target_to_device(target, device) for target in targets]
        losses, outputs = model(images, targets)

        model_time_avg += time.time() - model_time
        
        loss = sum([x for x in losses.values()])
        for k,l in losses.items():
            test_avg_losses[k] += l.item()
    
    model_time_avg = len(data_loader)
    
    test_avg_losses = {k: loss / len(data_loader) for k, loss in test_avg_losses.items()} #average loss
    test_avg_loss = sum([x for x in test_avg_losses.values()]) # total loss
    
    #save data into the logger
    for k, loss in test_avg_losses.items():
        logger.add_scalar('val_' + k, loss, epoch)
    logger.add_scalar('val_avg_loss', test_avg_loss, epoch)
    logger.add_scalar('model_time', model_time_avg, epoch)
    
    return test_avg_loss
    
    
def train_poses(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        
        lr_scheduler = None,
        
        batch_size = 16,
        n_epochs = 2000,
        num_workers = 1,
        save_frequency = 200
        ):
    
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            collate_fn = collate_to_dict,
                            pin_memory = True
                            )

    val_loader = DataLoader(val_flow, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn = collate_to_dict,
                            pin_memory = True
                            )

    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        train_one_epoch(save_prefix, 
                         model, 
                         optimizer, 
                         lr_scheduler, 
                         train_loader, 
                         device, 
                         epoch, 
                         logger
                         )
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        test_avg_loss = evaluate_one_epoch(save_prefix, 
                           model, 
                           val_loader, 
                           device, 
                           epoch, 
                           logger
                           )
        
        
        
        desc = 'epoch {} , val_loss={}'.format(epoch, test_avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'train_flow_input_parameters': train_loader.dataset.input_parameters,
                'val_flow_input_parameters': train_loader.dataset.input_parameters
            }
        
        if hasattr(model, 'input_parameters'):
            state['model_input_parameters'] = model.input_parameters
        
        
        is_best = test_avg_loss < best_loss
        if is_best:
            best_loss = test_avg_loss  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 
    
        