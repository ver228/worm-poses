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
from worm_poses.trainer import _get_flow
from worm_poses.flow import maps2skels

from skimage.feature import peak_local_max

if __name__ == '__main__':
    n_segments = 49
    #model_path = '/Users/avelinojaver/workspace/WormData/results/worm-poses/logs/20190107_183344_CPM_adam_lr0.0001_wd0.0_batch48/model_best.pth.tar'
    #model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190108_120557_CPM_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
    #model = CPM(n_segments = n_segments)
    #out_size = 40
    
#    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190109_185230_CPMout_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
#    model = CPM(n_segments = n_segments, same_output_size = True)
#    dataset = 'pooled'
#    out_size = 160
    
#    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/manually-annotated_20190115_092358_CPMout_adam_lr0.0001_wd0.0_batch24/checkpoint.pth.tar'
#    model = CPM(n_segments = n_segments, same_output_size = True)
#    dataset = 'manually-annotated'
#    out_size = 160
    
    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/manually-annotated_20190115_140602_CPMout_adam_lr0.0001_wd0.0_batch24/model_best.pth.tar'
    n_segments = 25
    model = CPM(n_segments = n_segments, same_output_size = True)
    dataset = 'manually-annotated'
    out_size = 160
   
    
    
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    
    gen = _get_flow(dataset, out_size)
    
    #gen.fold_skeleton = False
    #gen = SkeletonMapsFlow(out_size = out_size)
    #gen = SkeletonMapsFlow(only_unskeletonized = True)
    
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
    
    
    criterion = CPMLoss()
    
    
    #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
    with torch.no_grad():
        X = X.to(device)
        target = target.to(device)
        
        outs = model(X)
    
    
    res = outs[-1]
    #%%
    def _get_peaks(cpm_maps):
        all_coords = []
        for mm in cpm_maps:
            coords = peak_local_max(mm, threshold_abs = 0.1)
            
            all_coords.append(coords)
        return all_coords
    
    #%%
    for roi, skel_map_t, skel_map_r in zip(X, target, res):
        roi = roi.squeeze(dim=0)
        
        fig, axs = plt.subplots(1,3, figsize = (15, 5), sharex = True, sharey = True)
        axs[0].imshow(roi)
        
        skeletons_r = maps2skels(skel_map_r.numpy())
        skeletons_t = maps2skels(skel_map_t.numpy())
        
        
        mid = 24
        plt.figure()
        axs[1].imshow(roi, cmap = 'gray')
        for ss in skeletons_t:
            axs[1].plot(ss[:, 0], ss[:, 1])
            axs[1].plot(ss[mid, 0], ss[mid, 1], 'o')
        plt.axis('off')
        
        
        plt.figure()
        axs[2].imshow(roi, cmap = 'gray')
        for ss in skeletons_r:
            axs[2].plot(ss[:, 0], ss[:, 1], '.-')
            axs[2].plot(ss[mid, 0], ss[mid, 1], 'o')
        plt.axis('off')
        
#        all_coords = _get_peaks(skel_map_t.numpy())
#        cc = np.concatenate(all_coords)
#        cc_midbody = all_coords[-1]
#        
#        axs[1].imshow(roi, cmap = 'gray')
#        axs[1].plot(cc[:, 1], cc[:, 0], '.r')
#        axs[1].plot(cc_midbody[:, 1], cc_midbody[:, 0], 'ob')
#        
#        
#        all_coords = _get_peaks(skel_map_r.numpy())
#        cc = np.concatenate(all_coords)
#        cc_midbody = all_coords[-1]
#        
#        axs[2].imshow(roi, cmap = 'gray')
#        axs[2].plot(cc[:, 1], cc[:, 0], '.r')
#        axs[2].plot(cc_midbody[:, 1], cc_midbody[:, 0], 'ob')
        
        
        
    #%%
#    for roi, skel_map_t, skel_map_r in zip(X, target, res):
#        roi = roi.squeeze(dim=0)
#        
#        fig, axs = plt.subplots(1,2)
#        
#        
#        for ii, ss in enumerate([skel_map_t, skel_map_r]):
#            axs[ii].imshow(roi)
#            skel_maps = ss.numpy()
#        
#            coords = []
#            for mm in skel_maps:
#                yr, xr = np.unravel_index(mm.argmax(), mm.shape)
#                coords.append((xr, yr))
#            
#            coords = np.array(coords)*160/out_size
#            axs[ii].plot(coords[:, 0], coords[:, 1], 'r-')
#            axs[ii].plot(coords[0, 0], coords[0, 1], 'ro')
#            
        
        
    #%%
#        il, ir = max(0, yr-2), min(mm.shape[0], yr+3)
#        jl, jr = max(0, xr-2), min(mm.shape[1], xr+3)
#        
#        
#        xx, yy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
#        vv = mm[il:ir, jl:jr]/mm[yr, xr]
#        
#        
#        yr += (xx*vv).mean()
#        xr += (yy*vv).mean()   
    
    