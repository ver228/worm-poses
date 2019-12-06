#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:25:26 2019

@author: avelinojaver
"""
import os
from pathlib import Path

pretrained_path = str(Path.home() / 'workspace/pytorch/pretrained_models/')
os.environ['TORCH_HOME'] = pretrained_path

import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import math

    
def _conv3x3(n_in, n_out):
    return [nn.Conv2d(n_in, n_out, 3, padding=1),
    nn.LeakyReLU(negative_slope=0.1, inplace=True)]

def _conv_layer(ni, nf, ks=3, stride=1, dilation=1):
    if isinstance(ks, (float, int)):
        ks = (ks, ks)
    
    if isinstance(dilation, (float, int)):
        dilation = (dilation, dilation)
    
    pad = [x[0]//2*x[1] for x in zip(ks, dilation)]
    
    return nn.Sequential(
           nn.Conv2d(ni, nf, ks, bias = False, stride = stride, padding = pad, dilation = dilation),
           nn.BatchNorm2d(nf),
           nn.ReLU(inplace = True)
           )

class PretrainedBackBone(nn.Module):
    
    def __init__(self, backbone_name = 'vgg19', pretrained = True):
        super().__init__()
        if backbone_name == 'vgg19':
            features = torchvision.models.vgg19(pretrained = pretrained).features[:18]
        elif backbone_name == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained = pretrained)
            features = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1
                    )
        else:
            ValueError(f'Not implemented {backbone_name}')
        
        self.features = features
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        
        x = self.features(x)
        return x
#%%

class CPM_PAF(nn.Module):
    '''
    Convolutional pose machine + Part Affinity Fields
    https://arxiv.org/abs/1602.00134
    https://arxiv.org/pdf/1611.08050.pdf
    '''
    def __init__(self, 
                 n_stages = 6, 
                 n_segments = 25,
                 n_affinity_maps = 20,
                 squeeze_factor = 1,
                 same_output_size = False,
                 backbone = None,
                 is_PAF = True
                 ):
        
        super().__init__()
        
        self.n_segments = n_segments
        self.n_affinity_maps = n_affinity_maps
        self.squeeze_factor = squeeze_factor
        self.n_stages = n_stages
        self.same_output_size = same_output_size
        self.n_input_feats = 256
        self.is_PAF = is_PAF
        
        
        self._cpm_out = int(self.n_segments // squeeze_factor)
        n_stage_inputs = self.n_input_feats + self._cpm_out
        
        
        if self.is_PAF:
            self._paf_out = int(self.n_affinity_maps*2 // squeeze_factor)
            n_stage_inputs += self._paf_out
        
        
        if backbone is None:
            self.feats = nn.Sequential(*[
                _conv_layer(1, 64, ks=3, stride = 1),
                _conv_layer(64, 64, ks=3, stride = 1),
                _conv_layer(64, 128, ks=3, stride = 2),
                _conv_layer(128, 128, ks=3, stride = 1),
                _conv_layer(128, 256, ks=3, stride = 2),
                _conv_layer(256, 256, ks=3, stride = 1),
                _conv_layer(256, 256, ks=3, stride = 1),
            ])
        else:
            self.feats = backbone
        
        
        self.CPM_stage1 = self._first_stacked_convs(n_in = self.n_input_feats, 
                                       n_out = self._cpm_out, 
                                       n_core = 256)
        self.stages_ini = [self.CPM_stage1]
        
        
        
        self.CPM_stages = []
        for ii in range(1, self.n_stages):
            new_stage = self._stage_stacked_convs(n_in = n_stage_inputs, 
                                                   n_out = self._cpm_out,
                                                   n_core = 256)
            
            stage_name = 'CPM_stage{}'.format(ii+1)
            setattr(self, stage_name, new_stage)        
    
            self.CPM_stages.append(new_stage)
        self.stages = [self.CPM_stages]
        
        
        if self.same_output_size:
            self.CPM_upscaling = self._upscaling_convs(n_in = self._cpm_out,
                                                       n_out = self.n_segments,
                                                       n_core = self.n_segments)
            
            self.upscalings = [self.CPM_upscaling]
        
        if self.is_PAF:
            self.PAF_stage1 = self._first_stacked_convs(n_in = self.n_input_feats, 
                                           n_out = self._paf_out, 
                                           n_core = 256)
            
            self.PAF_stages = []
            for ii in range(1, self.n_stages):
                new_stage = self._stage_stacked_convs(n_in = n_stage_inputs, 
                                               n_out = self._paf_out,
                                               n_core = 256)
                
                stage_name = 'PAF_stage{}'.format(ii+1)
                setattr(self, stage_name, new_stage)        
        
                self.PAF_stages.append(new_stage)
                
            if self.same_output_size:
                self.PAF_upscaling = self._upscaling_convs(n_in = self._paf_out,
                                                           n_out = self.n_affinity_maps*2,
                                                           n_core = self.n_affinity_maps*2)
    
    
            self.stages_ini.append(self.PAF_stage1)
            self.stages.append(self.PAF_stages)
            self.upscalings.append(self.PAF_upscaling)
    
    
    
    
    def _first_stacked_convs(self, n_in, n_out, n_core):
         return nn.Sequential(
                _conv_layer(n_in, n_core, ks=3, stride = 1),
                _conv_layer(n_core, n_core, ks=3, stride = 1),
                _conv_layer(n_core, n_core, ks=3, stride = 1),
                _conv_layer(n_core, n_core//2, ks=1, stride = 1),
                _conv_layer(n_core//2, n_out, ks=1, stride = 1)
            )
    
    def _stage_stacked_convs(self, n_in, n_out, n_core):
        _ini_layer = [_conv_layer(n_in, n_core, ks=3, stride = 1)]
        _rep_layers = [_conv_layer(n_core, n_core, ks=3, stride = 1) for _ in range(4)]
        _out_layers = [_conv_layer(n_core, n_core//2, ks=1, stride = 1),
                    _conv_layer(n_core//2, n_out, ks=1, stride = 1)]
        
        _layers = _ini_layer + _rep_layers + _out_layers
        return nn.Sequential(*_layers)
    
    
    def _upscaling_convs(self, n_in, n_out, n_core):
        _layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
                *_conv3x3(n_in, n_core),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
                *_conv3x3(n_core, n_out)
                ]
        return nn.Sequential(*_layers)
        
    
    def forward(self, X):
        
        pad_inv_ = None
        if self.same_output_size:
            nn = 2**2
            ss = [math.ceil(x/nn)*nn - x for x in X.shape[2:]]
            pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
            
            #use pytorch for the padding
            pad_ = [x for d in pad_[::-1] for x in d] 
            pad_inv_ = [-x for x in pad_] 
            X = F.pad(X, pad_, 'reflect')
        
        feats = self.feats(X)
        
        
        
        
        
        prev_out = [func(feats) for func in self.stages_ini]
        
        outs = [prev_out]
        for stage in zip(*self.stages):
            x_in = torch.cat([feats, *prev_out], dim=1)
            prev_out = [func(x_in) for func in stage]
            outs.append(prev_out)
        
        
        if self.same_output_size:
            outs_upscaled = []
            
            for stage_out in outs:
                out_up = []
                for func, x in  zip(self.upscalings, stage_out):
                    x_up = func(x)
                    if pad_inv_ is not None:
                        x_up = F.pad(x_up, pad_inv_)
                    out_up.append(x_up)
                        
                outs_upscaled.append(out_up)
            
            outs = outs_upscaled
        
        
        if self.is_PAF:
            outs_n = []
            for (cpm_out, paf_out) in outs:
                _dims = paf_out.shape
                paf_out = paf_out.view(_dims[0], _dims[1]//2, 2, _dims[-2], _dims[-1])
                outs_n.append((cpm_out, paf_out))
            outs = outs_n
        
        else:
            #i want to flatten the output
            outs = [x[0] for x in outs]        
        
        return outs
    
#%%  

    
    

#%%
if __name__ == '__main__':
    from losses import CPM_PAF_Loss, CPM_Loss
    
    X = torch.rand([4, 1, 65, 128])
    target = (torch.rand([4, 25, 65, 128]), torch.rand([4, 20, 2, 65, 128]))
    
    mod = CPM_PAF(same_output_size = True, 
                  squeeze_factor = 1)
    criterion = CPM_PAF_Loss(is_maxlikelihood = True)
    
    
    outs = mod(X)
    loss = criterion(outs, target)
    loss.backward()
    #%%
    mod = CPM_PAF(same_output_size = True, 
                  is_PAF = False)
    
    criterion = CPM_Loss(is_maxlikelihood = True)
    
    outs = mod(X)
    loss = criterion(outs, target[0])
    loss.backward()
    