#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:43:23 2020

@author: avelinojaver
"""

from pathlib import Path
import pickle
import tqdm
import numpy as np
import random
import gzip

if __name__ == '__main__':
    
    percentile_w_ratio = 1
    widths_file = Path.home() / 'workspace/WormData/screenings/single_worm/median_widths.p'
    with open(widths_file, 'rb') as fid:
        dat = pickle.load(fid)
        lengths_data, widths_data = map(np.array, list(zip(*dat))[-2:])
        
        ratio_widths = np.percentile(widths_data/lengths_data[:, None], percentile_w_ratio, axis = 0)
        
    
    src_dir = Path.home() / 'OneDrive - Nexus365/worms/rois2select/v1/'
    save_dir = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/rois4training/'
    selected_images_dir = Path.home() / 'rois2select/v1/ROIs/_done/'
    
    fnames = list(selected_images_dir.rglob('*.png'))
    files_keys = {}
    for fname in fnames:
        key = str(fname).replace(str(selected_images_dir), '')
        
        bn, _, set_id = key[1:-4].rpartition('_')
        
        if set_id.startswith('clusters'):
            set_name = 'clusters'
        else:
            set_name = 'coils'
        roi_id = int(set_id[len(set_name):])
        
        if not bn in files_keys:
            files_keys[bn] = []
        files_keys[bn].append((set_name, roi_id))
    
    
    data2save = []
    for bn, valid_keys in tqdm.tqdm(files_keys.items()):
        src_file = src_dir / (bn + '_ROIs.p')
        
        with open(src_file, 'rb') as fid:
            src_data = pickle.load(fid)
        
        for set_name, roi_id in valid_keys:
            roi, skels, corner, frame = src_data[set_name][roi_id]
            
            L = np.linalg.norm(np.diff(skels, axis = 1), axis=2).sum(-1)
            widths = L[:, None]*ratio_widths[None]
            
            row = (roi, None, widths, skels, None, None, None)
            data2save.append(row)
    
    random.shuffle(data2save)
    
    n_test = 300
    n_val = 300
    
    all_data = dict(test = data2save[:n_test],
         validation = data2save[n_test:(n_test + n_val)],
         train = data2save[(n_test + n_val):])
    
    data_type = 'from_NNv1'
    for set_type, dat in all_data.items():
        save_name = save_dir / f'{data_type}_{set_type}.p.zip'
        with gzip.GzipFile(save_name, 'wb') as fid:
            fid.write(pickle.dumps(dat, 1))
    
    
    