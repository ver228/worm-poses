#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 07:20:43 2018

@author: avelinojaver
"""
from pathlib import Path

import os
import pandas as pd
import tables
import tqdm
import numpy as np
import cv2

import random
import math


TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='blosc',
    shuffle=True,
    fletcher32=True)
#%%

from tierpsy.helper.misc import remove_ext, RESERVED_EXT
from tierpsy.helper.params import read_unit_conversions
from tierpsy.analysis.ske_create.helperIterROI import getAllImgROI


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
#%%
def extract_rois(row, root_dir, roi_worm_size, roi_total_size):
    
    mask_file = root_dir / row['mask_prefix']
    feat_file = root_dir / row['feat_prefix']
    
    with pd.HDFStore(mask_file, 'r') as fid:
        full_data = fid.get_node('/full_data')[:]
        save_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
    
        #tot_frames = fid.get_node('/mask').shape[0]
        #frames2read = np.arange(0, tot_frames + 1, save_interval)
        frames2read = np.array([x*save_interval for x in range(full_data.shape[0])])
        
        assert frames2read.size == full_data.shape[0]
        masks = fid.get_node('/mask')[frames2read, :, :]
        
        stage_position_pix = None
        if '/stage_position_pix' in fid:
           dat = fid.get_node('/stage_position_pix')
           if dat.shape[0] >= frames2read.max():
               stage_position_pix = dat[frames2read, :]
           else:
               stage_position_pix = np.full((len(frames2read), 2), np.nan)
           
           
    _, (microns_per_pixel_f, _), _ = read_unit_conversions(str(feat_file), 'r')
    
    all_rois = []
    with pd.HDFStore(feat_file, 'r') as fid:
       
        trajectories_data = fid['/trajectories_data']
        
        trajectories_data = trajectories_data[trajectories_data['frame_number'].isin(frames2read)]
        skels_g = fid.get_node('/coordinates/skeletons')
        widths_g = fid.get_node('/coordinates/widths')
        
        
        
        skel_ids = trajectories_data['skeleton_id'].astype(np.int)
        skel_ids = skel_ids[skel_ids>=0]
        if len(skel_ids) == 0:
            return
        
        
        skeletons = skels_g[skel_ids]
        max_size = np.nanmax(np.max(skeletons, axis=1)-np.min(skeletons, axis=1))
        max_size = max_size/microns_per_pixel_f
        if np.isnan(max_size):
            return
        
        _scale = roi_worm_size/max_size
        new_img_shape = tuple([int(round(x*_scale)) for x in full_data.shape[1:]])
        full_data = np.array([cv2.resize(x, new_img_shape[::-1], interpolation=cv2.INTER_LINEAR) for x in full_data])
        masks = np.array([cv2.resize(x, new_img_shape[::-1], interpolation=cv2.INTER_LINEAR) for x in masks])
        
        trajectories_data['coord_x'] *=_scale
        trajectories_data['coord_y'] *=_scale
        traj_group_by_frame = trajectories_data.groupby('frame_number')
        
        
        w_mean = np.nanmean(widths_g, axis=0)
        w_mean *= _scale/microns_per_pixel_f
        
        for ii, current_frame in enumerate(frames2read):
            try:
                frame_data = traj_group_by_frame.get_group(current_frame)
            except KeyError:
                continue
            full_img = full_data[ii]
            mask_img = masks[ii]
            
            #dictionary where keys are the table row and the values the worms ROIs
            full_in_frame = getAllImgROI(full_img, frame_data, roi_total_size)
            mask_in_frame = getAllImgROI(mask_img, frame_data, roi_total_size)
            
            
            for irow in mask_in_frame.keys():
                roi_mask, corner = mask_in_frame[irow]
                if roi_mask.shape != (roi_total_size,roi_total_size):
                    #border roi, problematic, ignore it
                    continue
                
                roi_full, _ = full_in_frame[irow]
                
                row_data = frame_data.loc[irow]
                
                skeleton_id = int(row_data['skeleton_id'])
                if skeleton_id >=0:
                    skel = skels_g[skeleton_id]
                    skel /= microns_per_pixel_f
                    
                    if stage_position_pix is not None:
                        skel -= stage_position_pix[ii, None, :]
                    
                    skel *= _scale
                    skel -= corner[None, :]
                    
                    
                else:
                    skel = np.full((49, 2), np.nan)
                
                is_skeletonized = int(~np.any(np.isnan(skel)))
                
                dd = row['movie_id'], current_frame, int(row_data['worm_index_joined']), skeleton_id, is_skeletonized
                all_rois.append((dd, roi_mask, roi_full, skel, w_mean))
    
    if len(all_rois) ==0 :
        return
    
    rois_data, roi_masks, roi_fulls, skels, ws_mean = map(np.array, zip(*all_rois))
    rois_data = pd.DataFrame(rois_data.astype(np.int32), columns = ROI_DATA_COLS)
    
    return rois_data, roi_masks, roi_fulls, skels, ws_mean
#%%
    
def _add_is_train(experiments_data, frac_train):
    indeces, is_test = [], []
    for screen_name, screen_data in experiments_data.groupby('screening_name'):
        inds = screen_data.index.tolist()
        random.shuffle(inds)
        
        n_test = math.ceil((1-frac_train)*len(inds))
        
        is_test += n_test*[1] + (len(inds)-n_test)*[0]
        indeces += inds
    is_test = pd.Series(is_test, indeces)
    experiments_data['is_test'] = is_test
    return experiments_data
#%%
def initialize_file(save_name, experiments_data, roi_size, frac_train = 0.99):
    #divide data into train and test set
    experiments_data = _add_is_train(experiments_data, frac_train)
    
    
    #make sure the exp_data has the correct format for strings
    exp_data_r = experiments_data.to_records(index=False)
    dtypes = []
    for col in experiments_data.columns:
        ss = str(exp_data_r[col].dtype)
        
        if ss == 'object':
            ss = 'S{}'.format(experiments_data[col].map(lambda x: len(x)).max())
        dtypes.append((col, ss))
    dtypes = np.dtype(dtypes)
    exp_data_r = exp_data_r.astype(dtypes)
    
    roi_data_dtypes = np.dtype([(x,np.int32) for x in ROI_DATA_COLS])
    
    #create the new file
    with tables.File(str(save_name), 'w') as fid_samples:
        fid_samples.create_table('/',
                    "experiments_data",
                    exp_data_r,
                    filters = TABLE_FILTERS)
         
        fid_samples.create_table('/',
                    "roi_data",
                    roi_data_dtypes,
                    filters = TABLE_FILTERS)
        
        coords_g = fid_samples.create_group('/', 'coordinates')
        fid_samples.create_earray(coords_g, 
                        'skeletons',
                        atom = tables.Float32Atom(),
                        shape = (0, 49, 2),
                        chunkshape = (1, 49, 2),
                        filters = TABLE_FILTERS
                        )
        
        fid_samples.create_earray(coords_g, 
                        'widths',
                        atom = tables.Float32Atom(),
                        shape = (0, 49),
                        chunkshape = (1, 49),
                        filters = TABLE_FILTERS
                        )
        
        fid_samples.create_earray('/', 
                        'mask',
                        atom = tables.Float32Atom(),
                        shape = (0, roi_size, roi_size),
                        chunkshape = (1, roi_size, roi_size),
                        filters = TABLE_FILTERS
                        )
        
        fid_samples.create_earray('/', 
                        'full_data',
                        atom = tables.Float32Atom(),
                        shape = (0, roi_size, roi_size),
                        chunkshape = (1, roi_size, roi_size),
                        filters = TABLE_FILTERS
                        )
        
        
#%%
if __name__ == '__main__':
    from multiprocessing import Pool
    save_name = Path.home() / 'workspace/WormData/results/data/roi_samples.hdf5'
    
    #save_name = Path.home() / 'OneDrive - Nexus365/worms/test_roi_samples.hdf5'
    #save_name = Path.home() / 'OneDrive - Nexus365/worms/roi_samples.hdf5'
    
    roi_worm_size = 128
    roi_total_size = roi_worm_size + 32
    root_dir = Path.home() / 'workspace/WormData/screenings'
    is_resume = False
    
    batch_size = 6
    def _process_row(row):
        return extract_rois(row, root_dir, roi_worm_size, roi_total_size)
    
    if not is_resume:
        exp_data = get_all_files(root_dir)
        initialize_file(save_name, exp_data, roi_total_size)
    
    
    with pd.HDFStore(save_name, 'r+') as fid:
        exp_data = fid['/experiments_data']
        
        roi_data = fid['/roi_data']
        #remove files that where already saved
        exp_data = exp_data[~exp_data['movie_id'].isin(roi_data['movie_id'].unique())]
        
        roi_data_g = fid.get_node('/roi_data')
        skels_g = fid.get_node('/coordinates/skeletons')
        widths_g = fid.get_node('/coordinates/widths')
        masks_g = fid.get_node('/mask')
        full_g = fid.get_node('/full_data')
        
        all_rows =[x for _, x in exp_data.iterrows()]
        
        
        for ind in tqdm.tqdm(range(0, len(all_rows), batch_size)):
            rows = all_rows[ind:ind+batch_size]
                   
            with Pool(batch_size) as p:
                outs = p.imap_unordered(_process_row, rows)
            
                for _out in outs:
                    if _out is None:
                        continue
                    
                    rois_data, roi_masks, roi_fulls, skels, w_mean = _out
                    
                    roi_data_g.append(rois_data.to_records(index=False))
                    
                    skels_g.append(skels.astype(np.float32))
                    widths_g.append(w_mean.astype(np.float32))
                    
                    masks_g.append(roi_masks.astype(np.float32))
                    full_g.append(roi_fulls.astype(np.float32))
                    
            
    cols = [ 'worm_index_joined', 'frame_number', 'coord_x', 'coord_y', 'threshold', 'roi_size', 'skeleton_id']
    with tables.File(str(save_name), 'r+') as fid:
        N = fid.get_node('/mask').shape[0]
        rr = roi_total_size//2
        x_dtypes = np.dtype([(x,np.int) for x in cols])
        X_r = np.array([(x, x, rr, rr, -1, roi_total_size, x) for x in range(N)], dtype=x_dtypes)
        fid.create_table('/',
                    "trajectories_data",
                    X_r,
                    filters = TABLE_FILTERS)
        
        
#    #%%
#    with pd.HDFStore(save_name, 'r') as fid:
#        roi_data = fid['/roi_data']
#        experiments_data = fid['/experiments_data']
#    
#    experiments_data = _add_is_train(experiments_data, 0.99)
#    
#    exp_data_r = experiments_data.to_records(index=False)
#    dtypes = []
#    for col in experiments_data.columns:
#        ss = str(exp_data_r[col].dtype)
#        
#        if ss == 'object':
#            ss = 'S{}'.format(experiments_data[col].map(lambda x: len(x)).max())
#        dtypes.append((col, ss))
#    dtypes = np.dtype(dtypes)
#    exp_data_r = exp_data_r.astype(dtypes)
#    #%%
#    with tables.File(save_name, 'r+') as fid:
#        fid.remove_node('/', 'experiments_data')
#        fid.create_table('/',
#                    "experiments_data",
#                    exp_data_r,
#                    filters = TABLE_FILTERS)
    