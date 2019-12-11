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


from worm_poses.models import CPM_PAF
from worm_poses.flow import maps2skels

#%%
if __name__ == '__main__':
    bn = 'allPAF/allPAF_PAF+CPM_20190611_162531_adam_lr0.0001_wd0.0_batch8'
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/' / bn / 'checkpoint.pth.tar'
    
    n_segments = 25
    n_affinity_maps = 20
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    
    
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True)
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies')
    #fnames = list(root_dir.glob('*.hdf5'))
    
    fnames = list(root_dir.glob('CX11314_Ch1_04072017_103259.hdf5'))
    
    
    #fnames = ['/Users/avelinojaver/Downloads/recording61.2r_X1.hdf5']
    #fnames = ['/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/raw/manual_annotations/raw/Phase1/MaskedVideos/NIC199_worms10_food1-10_Set7_Pos4_Ch4_19052017_153012.hdf5']
    for mask_file in tqdm.tqdm(fnames):
        mask_file = Path(mask_file)
        
        
        with tables.File(mask_file, 'r') as fid:
            img = fid.get_node('/mask')[10000]
        
        img = img.astype(np.float32)/255.
        #img = img[:1024, :1024]
        
        X = img[None, None]
        X = torch.tensor(X)
        
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            cpm_maps_r, paf_maps_r = outs[-1]
        
        
        
        cpm_map = cpm_maps_r[0]
        
        cpm_map = cpm_map.numpy()
        skeletons_r = maps2skels(cpm_map, threshold_abs = 0.05)
        
        mid = 25
        plt.figure()
        
        plt.imshow(img, cmap='gray')
        for ss in skeletons_r:
            plt.plot(ss[:, 0], ss[:, 1], '.-')
            plt.plot(ss[mid, 0], ss[mid, 1], 'o')
        break
        #%%
        