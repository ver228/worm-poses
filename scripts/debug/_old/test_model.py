#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
from worm_poses.models import CPM, CPMLoss

if __name__ == '__main__':
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    X = torch.rand([4, 1, 160, 160]).to(device)
    target = torch.rand([4, 49, 40, 40]).to(device)
    
    mod = CPM()
    mod = mod.to(device)
    
    criterion = CPMLoss()
    
    
    outs = mod(X)
    
    for x in outs:
        print(x.type())
    
    loss = criterion(outs, target)
    loss.backward()