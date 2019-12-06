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

from worm_poses.trainer import _get_flow, _get_model
from worm_poses.flow import maps2skels

from skimage.feature import peak_local_max


if __name__ == '__main__':
    
    #model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/manually-annotated-PAF_20190115_200956_CPMout-PAF_adam_lr0.0001_wd0.0_batch24/checkpoint.pth.tar'
    #n_affinity_maps = 4
    
    
    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'
    model_name = 'CPMout-PAF'
    dataset = 'manually-annotated-PAF'
    
    n_segments = 25
    n_affinity_maps = 20
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    
    
    model, out_size = _get_model(model_name, n_segments, n_affinity_maps)
    gen = _get_flow(dataset, out_size)
    gen.blank_patch_s = 0
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
    
    
    
    #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
    with torch.no_grad():
        X = X.to(device)
        outs = model(X)
    
    #%%
    cpm_maps_r, paf_maps_r = outs[-1]
    cpm_maps_t, paf_maps_t = target
    
    #%%
    def _get_peaks(cpm_maps):
        all_coords = []
        for mm in cpm_maps:
            coords = peak_local_max(mm, threshold_abs = 0.1)
            
            all_coords.append(coords)
        return all_coords
    
    
    #%%
   
    for ii, (roi, skel_map_t, skel_map_r) in enumerate(zip(X, cpm_maps_t, cpm_maps_r)):
        
        roi = roi.squeeze(dim=0)
        
        fig, axs = plt.subplots(1,3, figsize = (9, 3), sharex = True, sharey = True)
        axs[0].imshow(roi)
        
        skeletons_r = maps2skels(skel_map_r.numpy())
        skeletons_t = maps2skels(skel_map_t.numpy())
        
        
        mid = 24
        axs[1].imshow(roi, cmap = 'gray')
        for ss in skeletons_t:
            axs[1].plot(ss[:, 0], ss[:, 1])
            axs[1].plot(ss[mid, 0], ss[mid, 1], 'o')
        
        axs[2].imshow(roi, cmap = 'gray')
        for ss in skeletons_r:
            axs[2].plot(ss[:, 0], ss[:, 1], '.-')
            axs[2].plot(ss[mid, 0], ss[mid, 1], 'o')
        
        
        #%%
        
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
    def paf2rgb(paf_map):
        paf_map = paf_map.numpy()
        mm = np.arctan(paf_map[1], paf_map[0]) + np.pi/2
        mm[~np.any(paf_map!=0, axis=0)] = -0.1
        bb = mm + 0.1
        bb /= bb.max()
        
        return bb
        
        
    
    for roi, skel_map_t, skel_map_r in zip(X, paf_maps_t, paf_maps_r):
        roi = roi.squeeze(dim=0)
        
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(roi)
        
        #skel_m, _ = skel_map_t.max(dim=0)
        #axs[1].imshow(skel_m)
        
        #skel_m, _ = skel_map_r.max(dim=0)
        #axs[2].imshow(skel_m)
        
        ii = 0
        axs[1].imshow(paf2rgb(skel_map_t[ii]))
        axs[2].imshow(paf2rgb(skel_map_r[ii]))
        
    #%%
    
    
    
#        
#        
#        #axs[3].imshow(mm[ii])
#        
#        
#        
#        bb = mm[ii] + 0.1
#        bb /= bb.max()
#        
#        roi_rgb = roi.squeeze()[... , None].repeat(3, axis=2)
#        roi_rgb[..., 1][bb>0] = roi.max()
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
    
    