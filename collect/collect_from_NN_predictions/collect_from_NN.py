#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:10:35 2020

@author: avelinojaver
"""
import pandas as pd
import tables
import numpy as np
from pathlib import Path
import tqdm
import pickle
import matplotlib.pylab as plt


def get_hard_rois_single(mask_file, skels_file, min_traj_size = 500):
    rois2check = []
    with pd.HDFStore(skels_file, 'r') as fid:
        
        trajectories_data = fid['/trajectories_data']
        traj_g = trajectories_data.groupby('worm_index_joined').groups
        traj_g = [x for x in traj_g.items() if x[1].size > min_traj_size]
        traj_g = sorted(traj_g, key = lambda x : len(x[1]))
        
        for w_ind, inds in traj_g[::-1]:
            worm_data = trajectories_data.loc[inds]
            
            frames = worm_data['frame_number'].values
            frames2check, = np.where(np.diff(frames) > 1)
            rois2check.append(worm_data.iloc[frames2check])
            
    
    
    all_ROIs_coils = []
    if len(rois2check):
        rois2check = pd.concat(rois2check)
        
        is_stage_moving = None
        with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
            masks = fid_mask.get_node('/mask')
            skeletons = fid_skel.get_node('/coordinates/skeletons')
            
            if '/stage_position_pix' in fid_mask:
                stage_position_pix = fid_mask.get_node('/stage_position_pix')[:]
                is_stage_moving = np.isnan(stage_position_pix)
            
            
            for frame, frame_data in rois2check.groupby('frame_number'):
                if is_stage_moving is not None:
                    ini = max(0, frame-2)
                    if is_stage_moving[ini:frame+2].any():
                        continue
                
                img = masks[frame]
                
                for _, row in frame_data.iterrows():
                    roi_size = int(row['roi_size'])
                    half_roi = roi_size//2
                    xl = int(row['coord_x']) - half_roi
                    xl = min(max(0, xl), img.shape[1]-roi_size)
                    
                    yl = int(row['coord_y']) - half_roi
                    yl = min(max(0, yl), img.shape[0]-roi_size)
                    
                    roi = img[yl:yl+roi_size, xl:xl+roi_size]
                    
                    skel = skeletons[int(row['skeleton_id'])]
                    skel[:,0]-= xl
                    skel[:,1]-= yl
                    
                    all_ROIs_coils.append((roi, skel[None], (xl, yl), frame))
                    
                    # plt.figure()
                    # plt.imshow(roi, cmap = 'gray')
                    # plt.plot(skel[:,0], skel[:,1])
    
    def _get_cross_score(ss):
        
        L = np.linalg.norm(np.diff(ss, axis=0), axis=1).sum()
        ss = ss[::8]
        
        rr = np.linalg.norm((ss[None] - ss[:, None]), axis=-1)
        np.fill_diagonal(rr, 1e10)
        
        rr[rr < 5] = 10
        
        return rr.min()/L
    
    all_ROIs_coils = sorted(all_ROIs_coils, key = lambda x : _get_cross_score(x[1][0]))
    
    return all_ROIs_coils

if __name__ == '__main__':
    save_dir_root = Path.home() / 'workspace/WormData/worm-poses/rois2select/v1'
    
    
    screens_dir = Path.home() / 'workspace/WormData/screenings/'
    subdirs2check  = [#('Bertie_movies', 10),
                      ('single_worm', 3)
                      ]
    
    #f_subdirs = dict( masks = 'MaskedVideos', skelsNN = 'ResultsNN_v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32')
    f_subdirs = dict( masks = 'finished', skelsNN = 'ResultsNN')
    postfixes = dict( masks = '.hdf5', skelsNN = '_skeletonsNN.hdf5')
        
    
    for subdir, max_n_rois in subdirs2check:
        save_dir = save_dir_root / subdir 
        save_dir.mkdir(parents = True, exist_ok = True)
        
        #skels_file = mask_file.replace('.hdf5', '_skeletonsNN.hdf5')
    
        root_dir = screens_dir / subdir
        skel_files =  [x for x in (root_dir / f_subdirs['skelsNN']).rglob('*' + postfixes['skelsNN'])]
        
        for skels_file in tqdm.tqdm(skel_files, desc = f'{subdir} | Processing...'):
            
            mask_file = Path(str(skels_file).replace(f_subdirs['skelsNN'], f_subdirs['masks']).replace(postfixes['skelsNN'], postfixes['masks']))
            
            save_name = save_dir / (mask_file.stem + '_ROIs.p')
            
            if save_name.exists():
                continue
            
            assert mask_file.exists() and skels_file.exists()
            
            all_ROIs_coils = get_hard_rois_single(mask_file, skels_file)
            
            all_ROIs_coils = all_ROIs_coils[:max_n_rois]
            
            
            with open(save_name, 'wb') as fid:
                rois_d = {'coils':all_ROIs_coils}
                pickle.dump(rois_d, fid)