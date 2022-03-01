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
import torch
import time
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.data import SkelMapsFlow, collate_to_dict
from worm_poses.configs import read_flow_config

from torch.utils.data import DataLoader

import tqdm    
import matplotlib.pylab as plt

#%%

if __name__ == '__main__':
    

    tic = time.time()
    
    roi_size = 128
    _is_plot = True
    
    root_dir = '/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/worm-poses/rois4training_filtered'
    
    data_types = ['from_tierpsy', 'manual']
    set2read = 'validation'
    
    flow_args = read_flow_config('v5')
    gen = SkelMapsFlow(root_dir, **flow_args)
    
    val_loader = DataLoader(gen, 
                            batch_size = 8, 
                            num_workers = 4,
                            collate_fn = collate_to_dict
                            )
    #%%
    figsize = (40, 10)
    n_figs = 0
    
    for dat in tqdm.tqdm(val_loader):
        if _is_plot:
            #%%
            X, targets = dat
            for x, target in zip(X, targets):
                
                assert (target['skels']>=0).all()
                
                paf = target['PAF']
                
                assert x.shape == (1, gen.roi_size, gen.roi_size)
                
                
                fig, axs = plt.subplots(1, 4, figsize = figsize, sharex = True, sharey = True)
                n_figs += 1
                axs[0].imshow(x[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
                
                mid_ind = paf.shape[0]//2 + 1
                mm = (paf[mid_ind, 0]**2 + paf[mid_ind, 1]**2).sqrt()
                axs[-2].imshow(mm)
                
                mm, _ = (paf[:, 0]**2 + paf[:, 1]**2).sqrt().max(axis = 0)
                axs[-1].imshow(mm)
                
                for ss in target['skels']:
                    axs[1].plot(ss[:, 0], ss[:, 1], '.-', lw=3)
                    axs[1].plot(ss[mid_ind, 0], ss[mid_ind, 1], 'o', lw=3)
                    
                for ax in axs:
                    ax.axis('off')
                fig.savefig(f'S{n_figs}.png')
            if n_figs > 5:
                break
    #%%
    print(f'\nTotal Time {time.time() - tic}')
    
    if _is_plot:
        ss = [t['skels'] for t in targets]
        dat = [(torch.tensor(s.shape[0]*[ii]).fill_(ii), s[..., 0], s[..., 1]) for ii, s in enumerate(ss)]
        batch_ind, x_ind, y_ind = [torch.cat(x, dim = 0) for x in zip(*dat)]
        
        N, S = x_ind.shape
        x_ind = x_ind.reshape(-1)
        y_ind = y_ind.reshape(-1)
        batch_ind = batch_ind[:, None].expand((N, S)).reshape(-1)
        ch_ind = torch.arange(S)[None, :].expand(N, S).reshape(-1)
    
    
    
    