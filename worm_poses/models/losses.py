#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""

from torch import nn
import torch.nn.functional as F
import torch


def _targets2inds(targets):
    dat = []
    for ii, s in enumerate(targets):
        if s.shape[0]>0:
            y_ind = s[..., 1]
            x_ind = s[..., 0]
            batch_ind = torch.tensor(s.shape[0]*[ii]).fill_(ii)
            dat.append((batch_ind, x_ind, y_ind))
    
    
    if dat:
        batch_ind, x_ind, y_ind = [torch.cat(x, dim = 0) for x in zip(*dat)]
        
        N, S = x_ind.shape
        x_ind = x_ind.reshape(-1)
        y_ind = y_ind.reshape(-1)
        batch_ind = batch_ind[:, None].expand((N, S)).reshape(-1)
        ch_ind = torch.arange(S)[None, :].expand(N, S).reshape(-1)
        
        return batch_ind, ch_ind, y_ind, x_ind
    else:
        return [torch.zeros(0, dtype=torch.long)]*4


class LossWithBeliveMaps(nn.Module):
    def __init__(self, 
                 target_loss = F.mse_loss,  
                 gauss_sigma = 2., 
                 increase_factor = 1., 
                 is_regularized = False, 
                 device = None
                 ):
        super().__init__()
        
        assert isinstance(gauss_sigma, float)
        assert isinstance(is_regularized, bool)
        
        self.target_loss = target_loss
        self.increase_factor = increase_factor
        self.is_regularized = is_regularized
        
        gaussian_kernel = self.get_gaussian_kernel(gauss_sigma)
        self.gaussian_kernel = nn.Parameter(gaussian_kernel)
    
    @staticmethod
    def get_gaussian_kernel(gauss_sigma, device = None):
        #convolve with a gaussian filter
        kernel_size = int(gauss_sigma*4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        rr = kernel_size//2
    
        x = torch.linspace(-rr, rr, kernel_size, device = device)
        y = torch.linspace(-rr, rr, kernel_size, device = device)
        xv, yv = torch.meshgrid([x, y])
        
        # I am not normalizing it since i do want the center to be equal to 1
        gaussian_kernel = torch.exp(-(xv.pow(2) + yv.pow(2))/(2*gauss_sigma**2))
        return gaussian_kernel
    
    
    def targets2belivemaps(self, targets, expected_shape, device = None):
        masks = torch.zeros(expected_shape, device = device)
        
        batch_ind, ch_ind, y_ind, x_ind= _targets2inds(targets)
        masks[batch_ind, ch_ind, y_ind, x_ind] = 1.
        
        kernel_size = self.gaussian_kernel.shape[0]
        gauss_kernel = self.gaussian_kernel.expand(1, expected_shape[1], kernel_size, kernel_size)
        belive_map = F.conv2d(masks, gauss_kernel, padding = kernel_size//2)
        belive_map *= self.increase_factor
        
        return belive_map 
    
    def forward(self, prediction, target):
        target_map = self.targets2belivemaps(target, 
                                             prediction.shape,  
                                             device = prediction.device
                                             )
        loss = self.target_loss(prediction, target_map)
        
        return loss

class MaximumLikelihoodLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    
    PAF_factor = 0.001 #use a factor to make match it with the PAF loss
    def __init__(self):
        super().__init__()
    
    def _predictions2softmax(self, predictions):
        n_batch, n_out, w, h = predictions.shape
        predictions = predictions.contiguous()
        
        #apply the log_softmax along all the images in a given channel in order to include empty images
        pred_l = predictions.transpose(0, 1).contiguous().view(n_out, -1)
        pred_l = nn.functional.log_softmax(pred_l, dim= 1)
        pred_l = pred_l.view(n_out, n_batch, w, h).transpose(0, 1)
        return pred_l
    
    
    def forward(self, predictions, targets):
        pred_l = self._predictions2softmax(predictions)
        batch_ind, ch_ind, y_ind, x_ind= _targets2inds(targets)
        keypoints = pred_l[batch_ind, ch_ind, y_ind, x_ind]
        loss = -keypoints.mean()
        
        
        return self.PAF_factor*loss

class SymetricMLELoss(MaximumLikelihoodLoss):
    def _get_keypoints_w_switch(self, pred_l, targets):
        batch_ind, ch_ind, y_ind, x_ind = _targets2inds(targets)
        keypoints_normal = pred_l[batch_ind, ch_ind, y_ind, x_ind]
        
        B, C, H, W = pred_l.shape
        ch_ind_switched = (C-1) - ch_ind
        keypoints_switched = pred_l[batch_ind, ch_ind_switched, y_ind, x_ind]
        
        #targets_switched = [torch.flip(x, dims = (1,)) for x in targets]
        #keypoints_normal = self._prediction_keypoints(pred_l, targets)
        #keypoints_switched = self._prediction_keypoints(pred_l, targets_switched)
        #this values are negative since they are the log of a softmax
        #we want to maximize this values
        keypoints = torch.max(keypoints_normal, keypoints_switched)
        return keypoints
    
    def forward(self, predictions, targets, is_valid_ht = None):
        pred_l = self._predictions2softmax(predictions)
        
        if is_valid_ht is None:
            keypoints = self._get_keypoints_w_switch(pred_l, targets)
        else:
            skels_valid_ht = [s[ind] for s,ind in zip(targets, is_valid_ht)]
            skels2switch = [s[~ind] for s,ind in zip(targets, is_valid_ht)]
            
            batch_ind, ch_ind, y_ind, x_ind = _targets2inds(skels_valid_ht)
            keypoints1 = pred_l[batch_ind, ch_ind, y_ind, x_ind]
            
            
            keypoints2 = self._get_keypoints_w_switch(pred_l, skels2switch)
            
            keypoints = torch.cat((keypoints1, keypoints2))
            
        
        loss = -keypoints.mean()
        
        
        return self.PAF_factor*loss
    


class SymetricPAFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_seg = nn.MSELoss(reduction = 'none')
        self.criterion_bgnd = nn.MSELoss(reduction = 'mean')
        
        
    def forward(self, predictions, targets):
        t_mask = (targets != 0.) 
        
        ss = t_mask.any(dim=2)
        mask_segs = ss.unsqueeze(2).expand_as(targets)
        mask_bgnd = (~ss.any(dim=1))[:, None, None].expand_as(targets)
        
        predictions_switched = -torch.flip(predictions, dims = (1,))
        
        targets_segs = targets[mask_segs]
        mse_normal = self.criterion_seg(predictions[mask_segs], targets_segs)
        mse_switched = self.criterion_seg(predictions_switched[mask_segs], targets_segs)
        
        mse = torch.min(mse_normal, mse_switched)
        loss_seg = torch.mean(mse)
        
        targets_bgnd = targets[mask_bgnd]
        loss_bgnd = self.criterion_bgnd(predictions[mask_bgnd], targets_bgnd)
        loss = loss_seg + loss_bgnd
        
        # w1 = targets_segs.shape[0]
        # w2 = targets_bgnd.shape[0]
        # tot = w1+w2
        # loss = (w1/tot)*loss_seg + (w2/tot)*loss_bgnd
        
        return loss
        
