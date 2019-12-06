#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""

from .misc import log_dir_root_dflt, save_checkpoint


from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import tqdm

import math
import sys
from collections import defaultdict

from ..flow import collate_simple

def trainer_detector(basename,
        model,
        device,
        flow_train,
        optimizer,
        log_dir = log_dir_root_dflt,
        batch_size = 16,
        lr = 1e-4, 
        weight_decay = 0.0,
        n_epochs = 2000,
        num_workers = 1,
        init_model_path = None,
        save_frequency = 100
        ):
    
    
    train_loader = DataLoader(flow_train, 
                              batch_size = batch_size,
                              num_workers = num_workers,
                              collate_fn = collate_simple
                              )
    
    
    
    model = model.to(device)
    
    log_dir = log_dir / basename
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    
    for epoch in pbar_epoch:
        #train
        model.train()
        pbar = tqdm.tqdm(train_loader, desc = f'{basename} Train')        
        train_avg_loss = 0
        
        individual_losses = defaultdict(int)
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: [x.to(device) for x in v] if isinstance(v, list) else v.to(device) 
                                    for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
    
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
    
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                
                sys.exit(1)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_avg_loss += loss_value
            for k,v in loss_dict.items():
                individual_losses[k] += v.item()
                
        train_avg_loss /= len(train_loader)
        logger.add_scalar('train_loss', train_avg_loss, epoch)
        
        for k,v in individual_losses.items():
            logger.add_scalar('train_' + k, v/len(train_loader), epoch)
        
        
        avg_loss = train_avg_loss
        
        desc = f'loss={avg_loss}'
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
