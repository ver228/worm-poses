#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:38:28 2019

@author: avelinojaver
"""
import torch
from torch import nn

class MaximumLikelihoodLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        
        n_batch, n_out, w, h = pred.shape
        
        #I am pooling all the data in a batch from a given segment together in order to give similar weights to ROIs with multiple worms 
        target_l = target.transpose(0, 1).contiguous().view(n_out, -1)
        pred_l = pred.transpose(0, 1).contiguous().view(n_out, -1)
        pred_l = nn.functional.log_softmax(pred_l, dim= 1)
        
        ch_losses = -(pred_l*target_l).sum(dim = 1)/target_l.sum(dim = 1)
        loss = ch_losses.sum()
        
        return loss
    
class CPM_Loss(nn.Module):
    def __init__(self, is_maxlikelihood = False):
        super().__init__()
        
        if is_maxlikelihood:
            self.loss_func = MaximumLikelihoodLoss()
        else:
            self.loss_func = nn.MSELoss()
        
        
    def forward(self, outs, target):
        losses = []
        for cpm_out in outs:
            loss = self.loss_func(cpm_out, target)
            losses.append(loss)
        
        return sum(losses)

class CPM_PAF_Loss(nn.Module):
    def __init__(self, is_maxlikelihood = False):
        super().__init__()
        self.l2 = nn.MSELoss()
        
        self.is_maxlikelihood = is_maxlikelihood
        
        if self.is_maxlikelihood:
            self.alpha = 0.001 #use a factor to make both losses have a similar magnitud
            self.maxlikelihood = MaximumLikelihoodLoss()
        
    def forward(self, outs, target):
        pm_target, paf_target = target
        pose_map, PAFs = outs

        if self.is_maxlikelihood:
            cpm_loss = self.alpha*self.maxlikelihood(pose_map, pm_target)
        else:
            cpm_loss = self.l2(pose_map, pm_target)
        
        losses = [cpm_loss]
        
        for paf in PAFs:
            loss = self.l2(paf, paf_target)
            losses.append(loss)
        
        return sum(losses)

#%%
def normalize_softmax(xhat):
    
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = nn.functional.softmax(hh, dim = 2)
    hmax, _ = hh.max(dim=2)
    hh = hh/hmax.unsqueeze(2)
    hh = hh.view(n_batch, n_channels, w, h)
    
    return hh

def get_preeval_func(bn):
    if 'maxlikelihood' in bn:
        preeval_func = normalize_softmax
    else:
        preeval_func = lambda x : x
        
    return preeval_func

class OpenPoseCPMLoss(nn.Module):
    def __init__(self, is_maxlikelihood = False):
        super().__init__()
        self.l2 = nn.MSELoss()
        
        self.is_maxlikelihood = is_maxlikelihood
        
        if self.is_maxlikelihood:
            self.alpha = 0.001 #use a factor to make both losses have a similar magnitud
            self.maxlikelihood = MaximumLikelihoodLoss()
        
    def forward(self, outs, target):
        pm_target, paf_target = target
        pose_map, PAFs = outs
        
        
        if self.is_maxlikelihood:
            cpm_loss = self.alpha*self.maxlikelihood(pose_map, pm_target)
        else:
            cpm_loss = self.l2(pose_map, pm_target)
        
        losses = [cpm_loss]
        
        if isinstance(PAFs, torch.Tensor):
            loss = self.l2(PAFs, paf_target)
            losses.append(loss)
        else:
            for paf in PAFs:
                loss = self.l2(paf, paf_target)
                losses.append(loss)
            
        return sum(losses)