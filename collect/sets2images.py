#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:12:33 2020

@author: avelinojaver
"""
import cv2
import numpy as np
import matplotlib.pylab as plt
import itertools
from pathlib import Path
import gzip
import pickle

def _rgb2hex(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))[::-1]
    
prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = [_rgb2hex(x) for x in prop_cycle.by_key()['color']]
COLORS = itertools.cycle(COLORS)

def get_roi2save(roi, skels):
    roi2save = np.concatenate([roi, roi], axis=1)
    roi2save = cv2.cvtColor(roi2save, cv2.COLOR_GRAY2RGB)
    
    skels_i = skels.astype(np.int)
    for skel, col in zip(skels_i, COLORS):
        cv2.polylines(roi2save, [skel], isClosed = False, color = col)
        cv2.circle(roi2save, tuple(skel[0]), radius=2, color = col)
    return roi2save

if __name__ == '__main__':
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/')
    root_save_dir = Path.home() / 'Desktop/rois2check'
    
    data_type = 'manual-v2'
    for set2read in ['test', 'train', 'validation']:
    
        fname = root_dir / f'{data_type}_{set2read}.p.zip'
        with gzip.GzipFile(fname, 'rb') as fid:
            data_raw = pickle.load(fid)
        
        save_dir = root_save_dir / data_type / set2read
        save_dir.mkdir(exist_ok = True, parents = True)
        for ii, out in enumerate(data_raw):
            roi = out[1] if out[1] is not None else out[0]
            skels = out[3]
            roi2save = get_roi2save(roi, skels)
            
            save_name = save_dir / (f'{ii}.png')
            cv2.imwrite(str(save_name), roi2save)
            
            