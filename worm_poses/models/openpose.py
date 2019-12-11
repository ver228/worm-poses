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

import math
import torch
from torch import nn
import torch.nn.functional as F

vgg_cfg = dict(
    vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512],
    vgg13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512],
    vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512],
    vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512],
    
    )

def make_vgg_layers(in_channels = 3, vgg_type = 'vgg19', batch_norm=False):
    cfg = vgg_cfg[vgg_type]
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


class ConvPReLu(nn.Sequential):

    def __init__(self, in_channels, n_outputs, kernel_size = 3):
        padding = kernel_size // 2
        
        super().__init__(
                    nn.Conv2d(in_channels, n_outputs, kernel_size = kernel_size, padding = padding),
                    nn.PReLU()
                    )

class UpscalingBlock(nn.Module):
    def __init__(self, n_in, n_out, n_core = 32):
        super().__init__()
        #n_stage = int(math.ceil(math.log2(expected_magnification)))
        _layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
                   ConvPReLu(n_in, n_core),
                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
                   ConvPReLu(n_core, n_core)
                   ]
        
        self.core = nn.Sequential(*_layers)
        self.final_conv = ConvPReLu(n_core, n_out)
        
        
    def forward(self, X, target_size):
        X =  self.core(X)
        X = F.interpolate(X, target_size, mode = 'bilinear', align_corners=True) 
        X = self.final_conv(X)
        return X

class DenseConv(nn.Module):
    def __init__(self, n_inputs, n_feats):
        super().__init__()
        self.conv1 = ConvPReLu(n_inputs, n_feats)
        self.conv2 = ConvPReLu(n_feats, n_feats)
        self.conv3 = ConvPReLu(n_feats, n_feats)
        
    
        
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
        self.conv_out0 = ConvPReLu(n_3feats, 256, kernel_size = 1)
        self.conv_out1 = ConvPReLu(256, n_outputs, kernel_size = 1)
    
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
                 features_type = 'vgg19'
                 ):
        
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_stages = n_stages
        self.n_segments = n_segments
        self.n_affinity_maps = n_affinity_maps
         
        self.features = nn.Sequential(
                make_vgg_layers(n_inputs, vgg_type = features_type),
                ConvPReLu(512, 512),
                ConvPReLu(512, 256),
                ConvPReLu(256, 128)
                )


        paf_n_outs = 2*n_affinity_maps
        PAF_stages = []
        for i in range(n_stages - 1):
           n_in = 128  if i == 0 else 128 + paf_n_outs
           PAF_stages.append(CPMStage(n_in, paf_n_outs))
        self.PAF_stages = nn.ModuleList(PAF_stages)
        self.CPM_stage = CPMStage(128 + paf_n_outs, n_segments)
    
        self.PAF_upscale = UpscalingBlock(paf_n_outs, paf_n_outs)
        self.CPM_upscale = UpscalingBlock(n_segments, n_segments)
    
    def _adjust_PAF(self, paf, target_size):
        #upscale PAF only for the output 
        paf_out = self.PAF_upscale(paf, target_size)
        N, _, w, h = paf_out.shape
        paf_out = paf_out.reshape((N, self.n_affinity_maps, 2, w, h))
        return paf_out
      
    def forward(self, xin):
        feats = self.features(xin)
        
        # I am following the strategy of mask rcnn and interpolate data instead of undersample the target maps
        N, C, W, H = xin.shape
        
        
        paf = self.PAF_stages[0](feats)
        
        if self.training:
            PAFs = [paf]
            for paf_stage in self.PAF_stages[1:]:
                x = torch.cat((feats, paf), dim = 1)
                paf = paf_stage(x)
                PAFs.append(paf)
            PAFs = [self._adjust_PAF(x, (W, H)) for x in PAFs]
            assert len(PAFs) == self.n_stages - 1
        else:
            for paf_stage in self.PAF_stages[1:]:
                x = torch.cat((feats, paf), dim = 1)
                paf = paf_stage(x)
            PAFs = self._adjust_PAF(paf, (W, H))
        
        
        x = torch.cat((feats, paf), dim = 1)
        pose_map = self.CPM_stage(x)
        pose_map = self.CPM_upscale(pose_map, (W,H))
        
        return pose_map, PAFs
    
    
if __name__ == '__main__':
    #from losses import OpenPoseCPMLoss
    
    in_channels = 1
    n_batch = 4
    w, h = 65, 128
    
    
    n_stages = 4
    n_segments = 17
    n_affinity_maps = 15
    features_type = 'vgg11'
    
    X = torch.rand((n_batch, in_channels, w, h))
    
    target = (torch.rand([n_batch, n_segments, w, h]), torch.rand([n_batch, n_affinity_maps, 2, w, h]))
    
    #criterion = OpenPoseCPMLoss()
    model = OpenPoseCPM(n_stages = n_stages,
                        n_segments = n_segments,
                        n_affinity_maps = n_affinity_maps,
                        features_type = features_type,
                        )
    outs = model(X)
    
#    loss = criterion(outs, target)
#    loss.backward()
#    
#    model.eval()
#    outs = model(X)
#    assert all([x.shape == t.shape] for x, t in zip(outs, target))
#    
#    
#    import tqdm
#    X_big = torch.rand((n_batch, in_channels, 2048, 2048))
#    for i in tqdm.trange(1000):
#        outs = model(X)
    
    
    