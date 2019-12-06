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


from tierpsy.helper.misc import remove_ext, RESERVED_EXT


ROI_DATA_COLS = ['movie_id', 'frame', 'worm_index', 'skeleton_id', 'is_skeletonized']

#%%
def get_all_files(root_dir):
    
    invalid_ext = RESERVED_EXT.copy()
    invalid_ext.remove('_featuresN.hdf5')
    
    all_fnames = {}
    for exp_dir in root_dir.glob('*/'):
        
        exp_name = exp_dir.name
        if exp_name == 'Serena_Swarming':
            continue
        
        fnames = list(exp_dir.rglob('*.hdf5'))
        
        grouped_fnames = {}
        for fname in fnames:
            if not any([fname.name.endswith(x) for x in invalid_ext]) and 'Old_Results' not in str(fname):
                bn = remove_ext(fname.name)
                if not bn in grouped_fnames:
                    grouped_fnames[bn] = []
                grouped_fnames[bn].append(fname)
        
        #only keeps files with two files and sort them alphabetically so (video_file, feat_file)
        grouped_fnames = {k:sorted(v, key = lambda x : x.name) for k,v in grouped_fnames.items() if len(v) == 2}
        
        all_fnames[exp_name] = grouped_fnames
        
    root_dir_s = str(root_dir)
    root_dir_s = root_dir_s if root_dir_s.endswith(os.sep) else root_dir_s + os.sep
    
    exp_data = []
    for exp_name, grouped_fnames in all_fnames.items():
        for bn, (mask_file, feat_file) in grouped_fnames.items():
            mask_prefix = str(mask_file).replace(root_dir_s, '')
            feat_prefix = str(feat_file).replace(root_dir_s, '')
            
            exp_data.append((exp_name, bn, mask_prefix, feat_prefix))
            
    
    exp_data = pd.DataFrame(exp_data, columns = ['screening_name', 'base_name', 'mask_prefix', 'feat_prefix'])
    exp_data = exp_data.sort_values(by=['screening_name', 'base_name'])
    exp_data = exp_data.reset_index(drop=True)
    exp_data['movie_id'] = exp_data.index
    
    return exp_data

def _process_row(row):
    MIN_PIX_AREA = 500
    _negatives = []
    mask_file = root_dir / row['mask_prefix']
    feat_file = root_dir / row['feat_prefix']
    
    with pd.HDFStore(mask_file, 'r') as fid:
        full_data = fid.get_node('/full_data')
        save_interval = int(full_data._v_attrs['save_interval'])
    
        frames2read = np.array([x*save_interval for x in range(full_data.shape[0])])
        masks = fid.get_node('/mask')[frames2read, :, :]
        
    with pd.HDFStore(feat_file, 'r') as fid:
        trajectories_data_ori = fid['/trajectories_data']
        trajectories_data = trajectories_data_ori[trajectories_data_ori['frame_number'].isin(frames2read)]
    
    static = masks.min(axis=0)
    for _, t_row in trajectories_data.iterrows():
        r2 = t_row['roi_size'] // 2 + 1
        lx, rx = t_row['coord_x'] - r2, t_row['coord_x'] + r2
        lx, rx = int(max(0, lx)), int(rx)
        
        ly, ry = t_row['coord_y'] - r2, t_row['coord_y'] + r2
        ly, ry = int(max(0, ly)), int(ry)
        
        static[ly:ry, lx:rx] = 0
    
    
    dat2check = masks[0].copy()
    dat2check[static==0] = 0
    
    _, cnts, _ = cv2.findContours(dat2check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > MIN_PIX_AREA]
    
    
    for bb in bboxes:
        xl, xr = bb[0], bb[0] + bb[2]
        yl, yr = bb[1], bb[1] + bb[3]
        roi = dat2check[yl:yr, xl:xr].copy()
        _negatives.append(roi)
        
    return _negatives

def _process_row_skip_errors(row):
    try:
        return _process_row(row)
    except:
        return []
    
#%%
if __name__ == '__main__':
    from multiprocessing import Pool
    
    batch_size = 24
    
    root_dir = Path.home() / 'workspace/WormData/screenings'
    save_dir = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/from_tierpsy_negative'
    
    #root_dir = Path.home() / 'workspace/WormData/screenings/'
    #save_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/labelled_rois/from_tierpsy'
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)
    
    exp_data = get_all_files(root_dir)
    rows =[x for _, x in exp_data.iterrows()]
    #rows = rows[:100]
    
    all_negatives = []
    
    with Pool(batch_size) as p:
        outs = p.imap_unordered(_process_row_skip_errors, rows)
        
        pbar = tqdm.tqdm(outs, total=len(rows))
        for out in pbar:
            all_negatives += out
            pbar.set_description(f"Total collected: {len(all_negatives)}")
    
    save_name = save_dir / f'negative_from_tierpsy.p.zip'
    with gzip.GzipFile(save_name, 'wb') as fid:
        fid.write(pickle.dumps(all_negatives, 1))
    
        
    