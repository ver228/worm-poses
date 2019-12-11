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
    ss = [t['skels'] for t in targets]
    dat = [(torch.tensor(s.shape[0]*[ii]).fill_(ii), s[..., 0], s[..., 1]) for ii, s in enumerate(ss)]
    batch_ind, x_ind, y_ind = [torch.cat(x, dim = 0) for x in zip(*dat)]
    
    N, S = x_ind.shape
    x_ind = x_ind.reshape(-1)
    y_ind = y_ind.reshape(-1)
    batch_ind = batch_ind[:, None].expand((N, S)).reshape(-1)
    ch_ind = torch.arange(S)[None, :].expand(N, S).reshape(-1)
    
    return batch_ind, ch_ind, y_ind, x_ind


class LossWithBeliveMaps(nn.Module):
    def __init__(self, target_loss = F.mse_loss,  gauss_sigma = 2., increase_factor = 1., is_regularized = False, device = None):
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
        target_map = self.targets2belivemaps(target, prediction.shape, device = prediction.device)
        loss = self.target_loss(prediction, target_map)
        
        return loss

class MaximumLikelihoodLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        n_batch, n_out, w, h = predictions.shape
        predictions = predictions.contiguous()
        
        #apply the log_softmax along all the images in a given channel in order to include empty images
        pred_l = predictions.transpose(0, 1).contiguous().view(n_out, -1)
        pred_l = nn.functional.log_softmax(pred_l, dim= 1)
        pred_l = pred_l.view(n_out, n_batch, w, h).transpose(0, 1)
        
        
        batch_ind, ch_ind, y_ind, x_ind= _targets2inds(targets)
        
        keypoints = pred_l[batch_ind, ch_ind, y_ind, x_ind]
        
        loss = -keypoints.mean()
        
        #use a factor to make match it with the PAF loss
        return 0.001*loss
