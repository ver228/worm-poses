#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:51:15 2019

@author: avelinojaver
"""

shared_obj = {}

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
#mp.set_start_method('fork', force=True)
#mp.set_start_method('forkserver', force=True)

import sys
import numpy as np
import time
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.flow import SkelMapsSimpleFlow, read_data_files, collate_with_skels

from torch.utils.data import DataLoader

import tqdm    
import matplotlib.pylab as plt

import torch
#%%


if __name__ == '__main__':
    tic = time.time()
    
    roi_size = 128
    _is_plot = False
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    #root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
    #root_dir = '/tmp/avelino/worm-poses/'
    
    data_types = ['from_tierpsy', 'manual']
    set2read = 'validation'
    
    #val_data = read_data_files(root_dir = root_dir, set2read = 'validation')
    #shared_obj = {k: x.create_share_objs() for k,x in val_data.items()}
    
    
    gen = SkelMapsSimpleFlow(root_dir = root_dir,
                            set2read = set2read,
                             roi_size = roi_size, 
                             return_raw_skels = True,
                             return_affinity_maps = True,
                             is_fixed_width = True,
                             min_sigma = 1.,
                             width2sigma = -1,
                             fold_skeleton = False
                             )
    
    val_loader = DataLoader(gen, 
                            batch_size = 4, 
                            num_workers = 4,
                            collate_fn = collate_with_skels
                            )
#    #%%
    figsize = (10, 40)
    n_figs = 0
    for X,Y in tqdm.tqdm(val_loader):
        CPMs, PAFs, skels = Y
        
        if _is_plot:
            for x, cpm, paf, skels_true in zip(X, CPMs, PAFs, skels):
                
                #X,Y = gen[ind]
                assert x.shape == (1, gen.roi_size, gen.roi_size)
                assert not (np.isnan(cpm).any() or np.isnan(skels_true).any())
                
                
                fig, axs = plt.subplots(1, 4, figsize = figsize, sharex = True, sharey = True)
                n_figs += 1
                axs[0].imshow(x[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
                
                mm, _ = cpm.max(dim=0)
                axs[1].imshow(mm)
                
                axs[2].imshow(cpm[0])
                
                
                mm = (paf[0, 0]**2 + paf[0, 1]**2).sqrt()
                axs[3].imshow(mm)
                
                for ss in skels_true:
                    axs[0].plot(ss[:, 0], ss[:, 1], '.-')
                    axs[0].plot(ss[25, 0], ss[25, 1], 'o')
                    
                for ax in axs:
                    ax.axis('off')
                    
            if n_figs > 5:
                break
    #%%
    print(f'\nTotal Time {time.time() - tic}')