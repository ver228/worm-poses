#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 07:20:43 2018

@author: avelinojaver
"""
from pathlib import Path

import os
import pandas as pd
import tqdm
import numpy as np
import cv2

import gzip
import pickle
import random

    
#%%
if __name__ == '__main__':
    
    roi_size = 128
    n_rois_per_img = 40
    
    root_dir = Path.home() / 'Desktop/OneDrive_2020-01-28/bgnd/'
    save_dir = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/rois4training/'
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)
    
    all_negatives = []
    for fname in root_dir.glob('*.png'):
        img = cv2.imread(str(fname), -1)
        th = np.percentile(img, 75)
        
        rois_in_img = []
        
        max_trials = n_rois_per_img*4
        n_trials = 0
        while len(rois_in_img) < n_rois_per_img:
            n_trials += 1
            if n_trials > max_trials:
                break
            
            xl = random.randint(0, img.shape[0] - roi_size-1)
            yl = random.randint(0, img.shape[1] - roi_size-1)
            
            roi = img[xl:xl+roi_size, yl:yl+roi_size]
            assert roi.shape == (roi_size, roi_size)
            
            if np.percentile(roi, 25) < th:
                all_negatives.append(roi.copy())
            
    save_name = save_dir / f'negative_from_hydra-bgnd.p.zip'
    with gzip.GzipFile(save_name, 'wb') as fid:
        fid.write(pickle.dumps(all_negatives, 1))
    
    
    