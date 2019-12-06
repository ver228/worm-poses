#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:25:26 2019

@author: avelinojaver

Model based on :
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/62eebd63cab8db7e3c2f912ced309450fb08aef9/models/pose/body_25/pose_deploy.prototxt
    and 
    https://arxiv.org/pdf/1812.08008v2.pdf
"""

import torch
from torch import nn
import torch.nn.functional as F

_vgg19_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512]
def make_vgg_layers(in_channels = 3, cfg = _vgg19_cfg, batch_norm=False):
    #modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v


    return nn.Sequential(*layers)


def _conv_prelu(in_channels, n_outputs, kernel_size = 3):
    padding = kernel_size // 2
    
    return nn.Sequential(
                nn.Conv2d(in_channels, n_outputs, kernel_size = kernel_size, padding = padding),
                nn.PReLU()
                )

class DenseConv(nn.Module):
    def __init__(self, n_inputs, n_feats):
        super().__init__()
        self.conv1 = _conv_prelu(n_inputs, n_feats)
        self.conv2 = _conv_prelu(n_feats, n_feats)
        self.conv3 = _conv_prelu(n_feats, n_feats)
        
    
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return torch.cat((x1, x2, x3), dim = 1)
   
class CPMStage(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_feats = 96, n_blocks = 5):
        super().__init__()
        
        n_3feats = 3*n_feats
        
        blocks = []
        for i in range(n_blocks):
           n_in = n_inputs  if i == 0 else n_3feats
           blocks.append(DenseConv(n_in, n_feats))
           
        
        self.denseblocks = nn.ModuleList(blocks)
        self.conv_out0 = _conv_prelu(n_3feats, 256, kernel_size = 1)
        self.conv_out1 = _conv_prelu(256, n_outputs, kernel_size = 1)
    
    def forward(self, x):
        for block in self.denseblocks:
            x = block(x)
        x = self.conv_out0(x)
        x = self.conv_out1(x)
        return x
    
class OpenPoseCPM(nn.Module):
    def __init__(self, 
                 n_inputs = 1,
                 n_stages = 6, 
                 n_segments = 25,
                 n_affinity_maps = 20,
                 ):
        
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_stages = n_stages
        self.n_segments = n_segments
        self.n_affinity_maps = n_affinity_maps
        
        
        self.features = nn.Sequential(
                make_vgg_layers(n_inputs),
                _conv_prelu(512, 512),
                _conv_prelu(512, 256),
                _conv_prelu(256, 128)
                )


        paf_n_outs = 2*n_affinity_maps
        PAF_stages = []
        for i in range(n_stages - 1):
           n_in = 128  if i == 0 else 128 + paf_n_outs
           PAF_stages.append(CPMStage(n_in, paf_n_outs))
        self.PAF_stages = nn.ModuleList(PAF_stages)
        
        
        self.PM_stage = CPMStage(128 + paf_n_outs, n_segments)
    
    
    def _adjust_PAF(self, paf, W, H):
        #upscale PAF only for the output 
        paf_out = F.interpolate(paf, (W,H))
        paf_out = paf_out.reshape((-1, self.n_affinity_maps, 2, W, H))
        return paf_out
    

    
    def forward(self, xin):
        feats = self.features(xin)
        paf = self.PAF_stages[0](feats)
        
        PAFs = [paf]
        for paf_stage in self.PAF_stages[1:]:
            x = torch.cat((feats, paf), dim = 1)
            paf = paf_stage(x)
            PAFs.append(paf)
        x = torch.cat((feats, paf), dim = 1)
        pose_map = self.PM_stage(x)
        
        
        # I am following the strategy of mask rcnn and interpolate data instead of undersample the target maps
        N, C, W, H = xin.shape
        if self.training:
            PAFs = [self._adjust_PAF(x, W, H) for x in PAFs]
            assert len(PAFs) == self.n_stages - 1
        else:
            PAFs = self._adjust_PAF(PAFs[-1], W, H)
        pose_map = F.interpolate(pose_map, (W,H)) 
        
        return pose_map, PAFs
    
    
if __name__ == '__main__':
    from losses import OpenPoseCPMLoss
    
    in_channels = 1
    n_batch = 4
    w, h = 65, 128
    n_segments = 25
    n_affinity_maps = 20
    
    X = torch.rand((n_batch, in_channels, w, h))
    
    target = (torch.rand([n_batch, n_segments, w, h]), torch.rand([n_batch, n_affinity_maps, 2, w, h]))
    
    criterion = OpenPoseCPMLoss()
    model = OpenPoseCPM()
    
    outs = model(X)
    
    
    
    loss = criterion(outs, target)
    loss.backward()
    
    model.eval()
    outs = model(X)
    assert all([x.shape == t.shape] for x, t in zip(outs, target))
    
    
#    import tqdm
#    X_big = torch.rand((n_batch, in_channels, 2048, 2048))
#    for i in tqdm.trange(1000):
#        outs = model(X)
    
    
    