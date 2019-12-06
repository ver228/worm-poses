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
import cv2
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from worm_poses.models import CPM
from worm_poses.flow import maps2skels

from tierpsy.analysis.ske_create.helperIterROI import getROIfromInd

if __name__ == '__main__':
    n_segments = 49
    
    #model_path = '/Users/avelinojaver/workspace/WormData/results/worm-poses/logs/20190107_183344_CPM_adam_lr0.0001_wd0.0_batch48/model_best.pth.tar'
    #model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190108_120557_CPM_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
    #model = CPM(n_segments = n_segments)
    #out_size = 40
    
#    model_path = '/Volumes/rescomp1/data/WormData/results/worm-poses/logs/20190109_185230_CPMout_adam_lr0.0001_wd0.0_batch32/model_best.pth.tar'
#    model = CPM(n_segments = n_segments, same_output_size = True)
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
    
    
    #%%
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    #times2check = [5500, 8000, 10000, 12400, 15000, 17500, 22400]
    #times2check = [10000, 20000]
    times2check = [13000, 19000]
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/')
    
    for mask_file in root_dir.glob('*.hdf5'):
        #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/N2H_Ch1_01072017_100340.hdf5'
        #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch2_01072017_093003.hdf5'
        
        mask_file = Path(mask_file)
        feats_file = mask_file.parent / 'Results' / (mask_file.stem + '_featuresN.hdf5')
        
        
        with pd.HDFStore(str(feats_file), 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        
        X = []
        for frame_number in times2check:
            
            row, worm_roi, roi_corner  = getROIfromInd(str(mask_file), 
                                                       trajectories_data, 
                                                       frame_number, 
                                                       worm_index = 1, 
                                                       roi_size = -1)
            
            worm_roi = cv2.resize(worm_roi, (160, 160), interpolation=cv2.INTER_LINEAR)
            zeroes_mask = cv2.erode(worm_roi, kernel=np.ones((3,3))) == 0
            
            #worm_roi = worm_roi/255.
            valid_pix = worm_roi[~zeroes_mask]
            _scale = np.min(valid_pix), np.max(valid_pix)
            
            xx = (worm_roi.astype(np.float32)-_scale[0])/(_scale[1] - _scale[0])
            xx[zeroes_mask] = 1
             
            
            X.append(xx[None])
        
        
        X = torch.tensor(X).float()
        
        #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
        
        res = outs[-1]
        
        for roi, skel_map_r in zip(X, res):
            roi = roi.squeeze(dim=0)
            #%%
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
            #%%
            fig.suptitle(mask_file.stem)
        
        
    
    
    
    
    