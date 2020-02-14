#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:13:56 2020

@author: avelinojaver
"""

from pathlib import Path
import tqdm
import pickle
import cv2 
import itertools
import numpy as np
import matplotlib.pyplot as plt


def _rgb2hex(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))[::-1]
    
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = [_rgb2hex(x) for x in prop_cycle.by_key()['color']]
colors = itertools.cycle(colors)

def get_roi2save(roi, skels):
    roi2save = np.concatenate([roi, roi], axis=1)
    roi2save = cv2.cvtColor(roi2save, cv2.COLOR_GRAY2RGB)
    
    skels_i = skels.astype(np.int)
    for skel, col in zip(skels_i, colors):
        cv2.polylines(roi2save, [skel], isClosed = False, color = col)
        cv2.circle(roi2save, tuple(skel[0]), radius=2, color = col)
    return roi2save
    
    

if __name__ == '__main__':
    
    src_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/rois2select/v1/')
    target_dir = Path.home() / 'rois2select/v1/ROIs'
    
    #fnames = src_dir.rglob('Mating_Assay/*_ROIs.p')
    fnames = src_dir.rglob('single_worm/*_ROIs.p')
    #fnames = src_dir.rglob('*_ROIs.p')
    #%%
    for fname in tqdm.tqdm(fnames):
        save_prefix = str(fname).replace(str(src_dir), str(target_dir))[:-7]
        
        with open(fname, 'rb') as fid:
            data = pickle.load(fid)
        
        for (k, vals) in data.items():
            for roi_id, (roi, skels, _, _) in enumerate(vals):
                roi2save = get_roi2save(roi, skels)
                save_name = Path(f'{save_prefix}_{k}{roi_id}.png')
                save_name.parent.mkdir(parents = True, exist_ok = True)
                
                cv2.imwrite(str(save_name), roi2save)
             