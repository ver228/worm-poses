#!/usr/bin/env python3
# -*- coding: utf-8 -*-
f"""
Created on Wed Jun 26 13:01:22 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn
import torch

if __name__ == '__main__':
    #%%
    model = fasterrcnn_resnet50_fpn(num_classes = 4, 
                                    min_size = 512,
                                    max_size = 512,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.]
                                    )
    model.train()
    
    
    X = [torch.rand((3, 512, 512)), torch.rand((3, 512, 1024))]
    targets = [{
                'boxes': torch.tensor([[4, 5, 20, 30], [120, 125, 220, 230]]).float(), 
                'labels' : torch.tensor([1, 3])
                   }, 
                {
                 'boxes': torch.tensor([[400, 150, 450, 230]]).float(), 
                'labels' : torch.tensor([2])
                   }
                ]
    losses = model(X, targets = targets)
    
    