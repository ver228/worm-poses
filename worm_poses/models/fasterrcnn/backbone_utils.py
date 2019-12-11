#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:16:08 2019

@author: avelinojaver
"""

#modified from torchvision  I need to add the p6p7 layer option
#https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py

from collections import OrderedDict

import torch
from torch import nn

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops

class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, 
                 backbone, 
                 return_layers, 
                 in_channels_list, 
                 out_channels, 
                 last_level = 'pool'
                 ):
        
        
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        
        if last_level == 'pool':
            extra_blocks = LastLevelMaxPool()
        elif last_level == 'p6p7':
            extra_blocks = LastLevelP6P7(in_channels_list[-1] , out_channels)
        else:
            raise ValueError(f'Not implemented {last_level}.')
        
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks = extra_blocks,
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


class FrozenBatchNorm2dv2(misc_nn_ops.FrozenBatchNorm2d):
    def __init__(self, *args, **argkws):
        super().__init__( *args, **argkws)
        #the batchnorm in resnext is missing a variable in the presaved values, but the pytorch does not have it FrozenBatchNorm2d
        #so I am adding it
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))


def resnet_fpn_backbone(backbone_name, pretrained):
    print(backbone_name)
    
    norm_layer = FrozenBatchNorm2dv2 if 'resnext' in backbone_name else misc_nn_ops.FrozenBatchNorm2d
    
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    
    if backbone_name == 'resnet18' or backbone_name == 'resnet34':
        in_channels_stage2 = 64
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
    else:
        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, last_level = 'pool')

