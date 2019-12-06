#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import tqdm

from .misc import log_dir_root_dflt, save_checkpoint
from ..encoders import maps2skels, get_best_match, resample_curve, extrack_keypoints, join_skeletons_simple
from ..flow import collate_with_skels

def train_skeleton_maps(basename,
        model,
        device,
        flow_train,
        flow_val,
        criterion,
        optimizer,
        preeval_func = None,
        log_dir = log_dir_root_dflt,
        batch_size = 16,
        lr = 1e-4, 
        weight_decay = 0.0,
        n_epochs = 2000,
        num_workers = 4,
        init_model_path = None,
        save_frequency = 100
        ):
    
    cutoff_error = 2.
    
    if preeval_func is None:
        preeval_func = lambda x : x
        
    train_loader = DataLoader(flow_train, 
                              batch_size = batch_size,
                              num_workers = num_workers)
    
    val_loader = DataLoader(flow_val, 
                            batch_size = batch_size, 
                            num_workers = num_workers,
                            collate_fn = collate_with_skels
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
        for X, target in pbar:
            
            X = X.to(device)

            if isinstance(target, (list, tuple)):
                target = [x.to(device) for x in target]
            else:
                target = target.to(device)

            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            train_avg_loss += loss.item()
            
            
        
        train_avg_loss /= len(train_loader)
        logger.add_scalar('train_loss', train_avg_loss, epoch)
        
        with torch.no_grad():
            #train
            model.eval()
            pbar = tqdm.tqdm(val_loader, desc = f'{basename} Validation')        
            
            val_avg_loss = 0
            val_accuracy = 0
            tot_val = 0
            for X, Y in pbar:
                X = X.to(device)
                
                target = Y[:-1] if len(Y) > 2 else Y[0]
                #b_skels_true = Y[-1]
                
                if isinstance(target, (list, tuple)):
                    target = [x.to(device) for x in target]
                else:
                    target = target.to(device)
    
                pred = model(X)
                
                loss = criterion(pred, target)
                val_avg_loss += loss.item()
                
#                all_preds = []
#                
#                if isinstance(pred[-1], torch.Tensor):
#                    import pdb
#                    pdb.set_trace()
#                    cpm_maps_r, paf_maps_r = pred
#                    cpm_maps_r = preeval_func(cpm_maps_r)
#                    for cpm_map, paf_maps  in zip(cpm_maps_r, paf_maps_r):
#                        cpm_map = cpm_map.detach().cpu().numpy()
#                        paf_maps = paf_maps.detach().cpu().numpy()
#                        
#                        skels_pred =  maps2skels(cpm_map, paf_maps, _is_debug = False)
#                        all_preds.append(skels_pred)
#                
#                elif len(pred[-1]) == 2:
#                    #get skeletons
#                    cpm_maps_r, paf_maps_r = pred[-1]
#                    cpm_maps_r = preeval_func(cpm_maps_r)
#                    for cpm_map, paf_maps  in zip(cpm_maps_r, paf_maps_r):
#                        cpm_map = cpm_map.detach().cpu().numpy()
#                        paf_maps = paf_maps.detach().cpu().numpy()
#                        
#                        skels_pred =  maps2skels(cpm_map, paf_maps, _is_debug = False)
#                        all_preds.append(skels_pred)
#                        
#                        
#                else:
#                    cpm_maps_r = pred[-1]
#                    cpm_maps_r = preeval_func(cpm_maps_r)
#                    for cpm_map in cpm_maps_r:
#                        cpm_map = cpm_map.detach().cpu().numpy()
#                        keypoints = extrack_keypoints(cpm_map, threshold_relative = 0.25, threshold_abs = .25)
#                        skels_pred = join_skeletons_simple(keypoints, max_edge_dist = 10)
#                        all_preds.append(skels_pred)
#                
#                
#                for skels_pred, skels_true in zip(all_preds, b_skels_true):
#                    skels_pred = [resample_curve(x, skels_true[0].shape[0]) for x in skels_pred]
#                    closest_ind, closest_error = get_best_match(skels_pred, skels_true)
#                    is_valid = len(set(closest_ind)) == len(skels_true) #all the skeletons where selected
#                    is_valid = is_valid & all([x < cutoff_error for x in closest_error])# ... and all average distance between skeletons is less than 1
#                    val_accuracy += int(is_valid)
#                    tot_val += 1
                    
            val_avg_loss /= len(val_loader)
            
            logger.add_scalar('val_loss', val_avg_loss, epoch)
            
            if tot_val > 0:
                val_accuracy = val_accuracy / tot_val
                logger.add_scalar('val_acc', val_accuracy, epoch)
        
        avg_loss = val_avg_loss
        
        desc = f'loss={avg_loss}, acc={val_accuracy}'
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
