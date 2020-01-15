#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:57:12 2020

@author: avelinojaver
"""

import pickle
import gzip
import matplotlib.pylab as plt
import tqdm
import numpy as np

def is_invalid_with_zeros(roi, skels, max_zeros_frac = 0.1):
    skels_i = skels.astype('int')
        
    H,W = roi.shape
    scores = []
    for skel_i in skels_i:
        skel_i[skel_i[:, 0] >= W, 0] = W - 1
        skel_i[skel_i[:, 1] >= H, 1] = H - 1
        ss = roi[skel_i[..., 1], skel_i[..., 0]]
        scores.append((ss==0).mean())
    
    return any([x>max_zeros_frac for x in scores])

if __name__ == '__main__':
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/manual_train.p.zip'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/from_tierpsy_train.p.zip'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/from_tierpsy_validation.p.zip'
    
    with gzip.GzipFile(fname, 'rb') as fid:
        data_raw = pickle.load(fid)
    
    #%%
    for irow, row in enumerate(tqdm.tqdm(data_raw)):
        roi_mask, roi_full, widths, skels, contours, cnts_bboxes, clusters_bboxes = row
        
        
        
        if not is_invalid_with_zeros(roi_mask, skels):
            continue
        
        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
        for ax in axs:
            ax.imshow(roi_mask, cmap = 'gray')
            ax.axis('off')
        
        for skel in skels:
            plt.plot(skel[:, 0], skel[:, 1], '.-')
            plt.plot(skel[0, 0], skel[0, 1], 'vr')
            
        
        
        
        
        
        