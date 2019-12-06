#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:06:10 2019

@author: avelinojaver
"""
import pandas as pd
import tqdm
from pathlib import Path
import cv2
import os

from multiprocessing import Pool
import numpy as np


from tierpsy.analysis.ske_create.helperIterROI import getAllImgROI
from tierpsy.helper.misc import remove_ext, RESERVED_EXT
#from collect_sample_full import get_all_files

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

def extract_rois(row, root_dir, save_dir_mask, save_dir_full):
    
    mask_file = root_dir / row['mask_prefix']
    feat_file = root_dir / row['feat_prefix']
    
    with pd.HDFStore(mask_file, 'r') as fid_vid, pd.HDFStore(feat_file, 'r') as fid_feats:
        
        save_interval = int(fid_vid.get_node('/full_data')._v_attrs['save_interval'])
        
        
        frames2read = np.array([x*save_interval for x in range(fid_vid.get_node('/full_data').shape[0])])
        
        
        trajectories_data = fid_feats['/trajectories_data']
        trajectories_data = trajectories_data[trajectories_data['frame_number'].isin(frames2read)]
        trajectories_data = trajectories_data[trajectories_data['was_skeletonized'] == 0]
        
        if len(trajectories_data) == 0:
            return
        
        
        full_data = fid_vid.get_node('/full_data')[:]
        assert frames2read.size == full_data.shape[0]
        masks = fid_vid.get_node('/mask')[frames2read, :, :]
        
        
        
        
        traj_group_by_frame = trajectories_data.groupby('frame_number')
        for ii, current_frame in enumerate(frames2read):
            try:
                frame_data = traj_group_by_frame.get_group(current_frame)
            except KeyError:
                continue
            full_img = full_data[ii]
            mask_img = masks[ii]
            
            #dictionary where keys are the table row and the values the worms ROIs
            full_in_frame = getAllImgROI(full_img, frame_data, -1)
            mask_in_frame = getAllImgROI(mask_img, frame_data, -1)
            
            for irow in mask_in_frame.keys():
                roi_mask, corner = mask_in_frame[irow]
                
                roi_full, _ = full_in_frame[irow]
                
                w_id = frame_data.loc[irow, 'worm_index_joined']
                base_name = 'movie-{}_worm-{}_frame-{}.png'.format(row['movie_id'], w_id, current_frame)
                
                cv2.imwrite(str(save_dir_full / base_name), roi_full)
                cv2.imwrite(str(save_dir_mask / base_name), roi_mask)
#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/WormData/screenings'
    save_dir = Path.home() / 'workspace/WormData/results/unskeletonized'
    
    save_dir_full = save_dir / 'full'
    save_dir_mask = save_dir / 'mask'
    
    save_dir_full.mkdir(exist_ok=True, parents=True)
    save_dir_mask.mkdir(exist_ok=True, parents=True)
    
    
    exp_data = get_all_files(root_dir)
    exp_data.to_csv(str(save_dir / 'movies_info.csv'), index=False)
    
    batch_size = 16
    all_rows =[x for _, x in exp_data.iterrows()]
        
    def _process_row(row):
        return extract_rois(row, root_dir, save_dir_mask, save_dir_full)
        
    
    for ind in tqdm.tqdm(range(0, len(all_rows), batch_size)):
        rows = all_rows[ind:ind+batch_size]
        
        with Pool(batch_size) as p:
            outs = p.imap_unordered(_process_row, rows)
            for _ in outs:
                pass
        
    