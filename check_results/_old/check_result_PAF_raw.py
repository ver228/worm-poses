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
import tables


from worm_poses.trainer import _get_model
from worm_poses.flow import maps2skels

#%%
if __name__ == '__main__':
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
    
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    times2check = [13000, 19000]
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos')
    fnames = list(root_dir.glob('*.hdf5'))
    
    fnames = ['/Users/avelinojaver/Downloads/recording61.2r_X1.hdf5']
    for mask_file in tqdm.tqdm(fnames):
        mask_file = Path(mask_file)
        
        #%%
        with tables.File(mask_file, 'r') as fid:
            img = fid.get_node('/mask')[10000]
        
        img = img.astype(np.float32)/255.
        #%%
        #import cv2
        #img = cv2.resize(img, (1024, 1024))
        #img = img[:1024, :1024]
        
        X = img[None, None]
        X = torch.tensor(X)
        #%%
        #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            cpm_maps_r, paf_maps_r = outs[-1]
        
        
        #%%
        cpm_map = cpm_maps_r[0]
        
        cpm_map = cpm_map.numpy()
        skeletons_r = maps2skels(cpm_map, threshold_abs = 0.05)
        #%%
        mid = 25
        plt.figure()
        
        plt.imshow(img, cmap='gray')
        for ss in skeletons_r:
            plt.plot(ss[:, 0], ss[:, 1], '.-')
            plt.plot(ss[mid, 0], ss[mid, 1], 'o')
        
        #%%
        