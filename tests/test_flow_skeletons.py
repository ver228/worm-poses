#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:51:15 2019

@author: avelinojaver
"""

shared_obj = {}

import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
mp.set_start_method('fork', force=True)
#mp.set_start_method('forkserver', force=True)

import sys
import numpy as np
import time
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.flow import SkelMapsFlow, collate_to_dict, collate_simple

from torch.utils.data import DataLoader

import tqdm    
import matplotlib.pylab as plt

#%%

if __name__ == '__main__':
    flow_args = dict(
            samples_per_epoch = 400,
            data_types = ['from_tierpsy', 'manual'],
             negative_src = 'from_tierpsy_negative.p.zip',
             scale_int = (0, 255),
             
             roi_size = 256,
                 crop_size_lims = (50, 180),
                 negative_size_lims = (5, 180),
                 n_rois_lims = (1, 3),
                 int_expansion_range = (0.7, 1.3),
                 int_offset_range = (-0.2, 0.2),
                 blank_patch_range = (1/8, 1/3),
                 zoom_range = None,
                 
                 PAF_seg_dist = 6,
                 n_segments = 49
            )


    tic = time.time()
    
    roi_size = 128
    _is_plot = False
    
    #root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
    #root_dir = '/tmp/avelino/worm-poses/'
    
    data_types = ['from_tierpsy', 'manual']
    set2read = 'validation'
    
    #val_data = read_data_files(root_dir = root_dir, set2read = 'validation')
    #shared_obj = {k: x.create_share_objs() for k,x in val_data.items()}
    
    
    gen = SkelMapsFlow(root_dir, **flow_args)
    
    val_loader = DataLoader(gen, 
                            batch_size = 4, 
                            num_workers = 4,
                            collate_fn = collate_to_dict
                            )
    #%%
    figsize = (10, 40)
    n_figs = 0
    
    for dat in tqdm.tqdm(val_loader):
        if _is_plot:
            #%%
            X, targets = dat
            for x, target in zip(X, targets):
                
                assert (target['skels']>=0).all()
                
                paf = target['PAF']
                
                assert x.shape == (1, gen.roi_size, gen.roi_size)
                
                
                fig, axs = plt.subplots(1, 3, figsize = figsize, sharex = True, sharey = True)
                n_figs += 1
                axs[0].imshow(x[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
                
                ind = 25
                mm = (paf[ind, 0]**2 + paf[ind, 1]**2).sqrt()
                axs[1].imshow(mm)
                
                mm, _ = (paf[:, 0]**2 + paf[:, 1]**2).sqrt().max(axis = 0)
                axs[2].imshow(mm)
                
                for ss in target['skels']:
                    axs[0].plot(ss[:, 0], ss[:, 1], '.-')
                    axs[0].plot(ss[25, 0], ss[25, 1], 'o')
                    
                for ax in axs:
                    ax.axis('off')
                #%%
            if n_figs > 5:
                break
    #%%
    
    ss = [t['skels'] for t in targets]
    dat = [(torch.tensor(s.shape[0]*[ii]).fill_(ii), s[..., 0], s[..., 1]) for ii, s in enumerate(ss)]
    batch_ind, x_ind, y_ind = [torch.cat(x, dim = 0) for x in zip(*dat)]
    
    N, S = x_ind.shape
    x_ind = x_ind.reshape(-1)
    y_ind = y_ind.reshape(-1)
    batch_ind = batch_ind[:, None].expand((N, S)).reshape(-1)
    ch_ind = torch.arange(S)[None, :].expand(N, S).reshape(-1)
    
    
    
    
    
    
    #%%
    print(f'\nTotal Time {time.time() - tic}')