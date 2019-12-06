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
import tqdm
import torch
import matplotlib.pylab as plt
import numpy as np

from torch.utils.data import DataLoader

from worm_poses.models import CPM, CPMLoss
from worm_poses.flow import SkeletonMapsFlow

if __name__ == '__main__':
    n_segments = 49
    #model_path = '/Users/avelinojaver/workspace/WormData/results/worm-poses/logs/20190107_183344_CPM_adam_lr0.0001_wd0.0_batch48/model_best.pth.tar'
    #model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190108_120557_CPM_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
    #model = CPM(n_segments = n_segments)
    #out_size = 40
    
    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190109_185230_CPMout_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
    model = CPM(n_segments = n_segments, same_output_size = True)
    out_size = 160
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    #%%
    #gen = SkeletonMapsFlow()
    gen = SkeletonMapsFlow(out_size = out_size,
                           only_unskeletonized = True)
    #%%
    loader = DataLoader(gen, 
                        batch_size = 16, 
                        shuffle = True
                        )  
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    gen.test()
    for X, target in tqdm.tqdm(loader):
        break
    #%%
    criterion = CPMLoss()
    
    #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
    with torch.no_grad():
        X = X.to(device)
        target = target.to(device)
        
        outs = model(X)
    
    
    res = outs[-1]
    
    
    #%%
    
    for roi, skel_map_r in zip(X, res):
            roi = roi.squeeze(dim=0)
            
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(roi)
            
            
            skel_m, _ = skel_map_r.max(dim=0)
            axs[1].imshow(skel_m)
            
            
            coords = []
            for mm in skel_map_r:
                yr, xr = np.unravel_index(mm.argmax(), mm.shape)
                coords.append((xr, yr))
            
            coords = np.array(coords)*160/out_size
            
            axs[2].imshow(roi)
            axs[2].plot(coords[:, 0], coords[:, 1], 'r-')
            axs[2].plot(coords[0, 0], coords[0, 1], 'ro')
            