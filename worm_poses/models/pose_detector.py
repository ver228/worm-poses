#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:20:23 2019

@author: avelinojaver
"""

from .openpose import OpenPoseCPM
from .losses import MaximumLikelihoodLoss, LossWithBeliveMaps


import torch
from torch import nn
import torch.nn.functional as F


def _normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = F.softmax(hh, dim = 2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def get_loc_loss(loss_type):
    criterion = None
    if loss_type == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        preevaluation = _normalize_softmax
    else:
        parts = loss_type.split('-')
        loss_t = parts[0]
        
        gauss_sigma = [x for x in parts if x[0] == 'G'] #this must exists
        gauss_sigma = float(gauss_sigma[0][1:])
         
        increase_factor = [x for x in parts if x[0] == 'F'] #this one is optional
        increase_factor = float(increase_factor[0][1:]) if increase_factor else 1.
        
        
        is_regularized = [x for x in parts if x == 'reg']
        is_regularized = len(is_regularized) > 0
        
        
        preevaluation = lambda x : x
        if loss_t == 'l1smooth':
            target_loss = nn.SmoothL1Loss()
        elif loss_t == 'l2':
            target_loss = nn.MSELoss()
        elif loss_t == 'l1':
            target_loss = nn.L1Loss()
            
        criterion = LossWithBeliveMaps(target_loss, 
                                       gauss_sigma = gauss_sigma, 
                                       is_regularized = is_regularized,
                                       increase_factor = increase_factor
                                       )
    
    if criterion is None:
        raise ValueError(loss_type)
    return criterion, preevaluation


class BeliveMapsNMS(nn.Module):
    def __init__(self, threshold_abs = 0.0, threshold_rel = None, min_distance = 3):
        super().__init__()
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.min_distance = min_distance
        
    def forward(self, belive_map):
        kernel_size = 2 * self.min_distance + 1
        
        n_batch, n_channels, w, h = belive_map.shape
        hh = belive_map.contiguous().view(n_batch, n_channels, -1)
        max_vals, _ = hh.max(dim=2)
        
        x_max = F.max_pool2d(belive_map, kernel_size, stride = 1, padding = kernel_size//2)
        x_mask = (x_max == belive_map) #nms using local maxima filtering
        
        
        x_mask &= (belive_map > self.threshold_abs) 
        if self.threshold_rel is not None:
            vmax = max_vals.view(n_batch, n_channels, 1, 1)
            x_mask &= (belive_map > self.threshold_rel*vmax)
        
        
        outputs = []
        for xi, xm, xmax in zip(belive_map, x_mask, max_vals):
            ind = xm.nonzero()
            scores_abs = xi[ind[:, 0], ind[:, 1], ind[:, 2]]
            scores_rel = scores_abs/xmax[ind[:, 0]]
            
            skeletons = ind[:, [0, 2, 1]]
            outputs.append((skeletons, scores_abs, scores_rel))
        
        return outputs
    
    
class PoseDetector(nn.Module):
    def __init__(self, 
                 pose_loss_type = 'maxlikelihood',
                 
                 n_inputs = 1,
                 n_stages = 6, 
                 n_segments = 25,
                 n_affinity_maps = 20,
                 features_type = 'vgg19',
                 
                 nms_threshold_abs = 0.0,
                 nms_threshold_rel = 0.2,
                 nms_min_distance = 3,
                 
                 return_belive_maps = False
                 
                 ):
        
        
        super().__init__()
        
        _dum = set(dir(self))
        self.nms_threshold_abs = nms_threshold_abs
        self.nms_threshold_rel = nms_threshold_rel
        self.nms_min_distance = nms_min_distance
        
        self.pose_loss_type = pose_loss_type
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        self.mapping_network = OpenPoseCPM(n_inputs = n_inputs,
                             n_stages = n_stages, 
                             n_segments = n_segments,
                             n_affinity_maps = n_affinity_maps,
                             features_type = features_type
                             )
        
        self.cpm_criterion, self.preevaluation = get_loc_loss(pose_loss_type)
        self.paf_criterion = nn.MSELoss()
        
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_threshold_rel, nms_min_distance)
        
        self.return_belive_maps = return_belive_maps
    
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    
    def forward(self, x, targets = None):
        pose_map, PAFs = self.mapping_network(x)
        
        outputs = []
        
        if targets is not None:
            cpm_loss = self.cpm_criterion(pose_map, targets)
            
            target_PAFs = torch.stack([t['PAF'] for t in targets])
            if self.training:
                paf_loss = sum([self.paf_criterion(x, target_PAFs) for x in PAFs])
            else:
                paf_loss = self.paf_criterion(PAFs, target_PAFs)
            
            loss = dict(
                cpm_loss = cpm_loss,
                paf_loss = paf_loss
                )
            outputs.append(loss)
        
        if not self.training:
            xhat = self.preevaluation(pose_map)
            result = {}
#            outs = self.nms(xhat)
#            
#            result = []
#            for skeletons, scores_abs, scores_rel in outs:
#                
#                result.append(
#                    dict(
#                        skeletons = skeletons,
#                        scores_abs = scores_abs,
#                        scores_rel = scores_rel
#                        )
#                    )
            outputs.append(result)

        if self.return_belive_maps:
            outputs.append((xhat, PAFs))
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs
    
    