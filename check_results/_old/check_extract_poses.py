#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:01:46 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import tables
import numpy as np
import cv2

from tierpsy.analysis.traj_join.joinBlobsTrajectories import assignBlobTrajDF, joinGapsTrajectoriesDF
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='zlib',
    shuffle=True,
    fletcher32=True)

if __name__ == '__main__':
    #mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
    #mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5'
    
    
    #mask_file = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5')
    
    
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
    #mask_file = Path(mask_file)
    #skel_file = mask_file.parent / (mask_file.stem + 'skeletonsNN.hdf5')
    #skel_file = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818skelNN.hdf5')
    #skel_file_nn = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818_skeletons.hdf5')
    
    #skel_file = "/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818skelNN.hdf5"
    #skel_file = "/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839skelNN.hdf5"
    #skel_file = "/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/N2_worms10_CSCD068947_10_Set2_Pos5_Ch1_08082017_212337skelNN.hdf5"
    skel_file = "/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/results/JU2587_worms10_food1-10_Set1_Pos4_Ch1_20102017_125044skelNN.hdf5"
        
    skel_file_nn = skel_file.replace('skelNN.hdf5', '_skeletonsNN.hdf5')
    
    with pd.HDFStore(str(skel_file), 'r') as fid:
        skel_info = fid['/skel_info']
        
        skels = fid.get_node('/skeletons')[:]
        mid = skels.shape[1]//2
        
        coord_x, coord_y = skels[:, mid].T
        
        bounding_box_xmin, bounding_box_ymin = skels.min(axis = 1).T
        bounding_box_xmax, bounding_box_ymax = skels.max(axis = 1).T
        
        
        box_coords, box_wl, angles = zip(*[cv2.minAreaRect(ss) for ss in skels])
        box_length, box_width = zip(*[(x,y) if x > y else (y,x) for x,y in box_wl])
        
        tot = len(box_length)
        traj_df = pd.DataFrame({
                      'area' : np.ones(tot),
                      'coord_x' : coord_x,
                      'coord_y' : coord_y,
                      'box_length' : box_length, 
                      'box_width' : box_width, 
                      'bounding_box_xmin' : bounding_box_xmin,
                      'bounding_box_xmax' : bounding_box_xmax,
                      'bounding_box_ymin' : bounding_box_ymin,
                      'bounding_box_ymax' :bounding_box_ymax,
                      })
        
        traj_df = pd.concat((skel_info, traj_df), axis=1)
        #%%
        traj_df['worm_index_b'] = assignBlobTrajDF(traj_df, max_allowed_dist = 10, area_ratio_lim = (0, 1e3))
        #%%
        traj_df['worm_index'] = joinGapsTrajectoriesDF(traj_df, worm_index_type = 'worm_index_b', max_frames_gap = 25, area_ratio_lim = (0, 1e3))
        traj_df = traj_df[traj_df['worm_index']>=0]
        #%%
        min_track_size = 2#25
        displacement_smooth_win = 51
        
        
        trajectories_data = []
        skeletons_sorted = []
        
        worm_index = 1
        skeleton_id = 0
        for _, worm_data in traj_df.groupby('worm_index'):
            
            if len(worm_data) < min_track_size:
                continue
            
            t = worm_data['frame_number'].values.astype(np.int)
            
            bb_x = (worm_data['bounding_box_xmax'] - worm_data['bounding_box_xmin'] + 1).max()
            bb_y = (worm_data['bounding_box_ymax'] - worm_data['bounding_box_ymin'] + 1).max()
        
            roi_size = int(max(bb_x, bb_y) + 30)
            
            x = (worm_data['bounding_box_xmax'] + worm_data['bounding_box_xmin'])/2
            y = (worm_data['bounding_box_ymax'] + worm_data['bounding_box_ymin'])/2
            
            start_t, end_t = min(t), max(t)
            tnew = np.arange(start_t, end_t + 1, dtype=np.int32)
            fx = interp1d(t, x)
            fy = interp1d(t, y)
            xnew = fx(tnew)
            ynew = fy(tnew)
            
            
            track_size = len(tnew)
            if track_size > displacement_smooth_win and displacement_smooth_win > 3:
                xnew = savgol_filter(xnew, displacement_smooth_win, 3)
                ynew = savgol_filter(ynew, displacement_smooth_win, 3)
            
            worm_skels = np.full((len(tnew), 49,2), np.nan)
            
            t_offset = t-start_t
            
            worm_skels[t_offset] = skels[worm_data['skeleton_id']]
            #align skeletons
            #%
            t_offset = sorted(t_offset)
            for ii in range(len(t_offset)-1):
                ind1, ind2 = t_offset[ii], t_offset[ii+1]
                s1, s2 = worm_skels[ind1], worm_skels[ind2]
                s2_inv = s2[::-1]
                
                _pos = np.sum(np.abs(s1 - s2))
                _inv = np.sum(np.abs(s1 - s2_inv))
                
                if _pos > _inv:
                    worm_skels[ind2] = s2_inv
                    
                
                    
            worm_data_smooth = pd.DataFrame({
                    'worm_index_joined'  : [worm_index]*track_size,
                    'coord_x' : xnew,
                    'coord_y' : ynew,
                    'frame_number' : tnew,
                    'threshold' : -1,
                    'area' : -1,
                    'roi_size' : [roi_size]*track_size,
                    'skeleton_id' : np.arange(skeleton_id, skeleton_id+track_size),
                    'has_skeleton' : 1
                    })
        
            trajectories_data.append(worm_data_smooth)
            skeletons_sorted.append(worm_skels)
            
            assert len(worm_skels) == len(worm_data_smooth)
            
            skeleton_id += track_size
            worm_index += 1
            
#            if worm_index-1 == 13:
#                t1 = 5550
#                t2 = t1 + 20
#                
#                mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5'
#                with tables.File(mask_file, 'r') as fid:
#                    img = fid.get_node('/mask')[t1]
#                
#                plt.figure()
#                plt.imshow(img, cmap='gray')
#                plt.plot(worm_skels[t1, :, 0], worm_skels[t1, :, 1])
#                plt.plot(worm_skels[t1, 0, 0], worm_skels[t1, 0, 1], 'o')
#                plt.plot(worm_skels[t2, :, 0], worm_skels[t2, :, 1])
#                plt.plot(worm_skels[t2, 0, 0], worm_skels[t2, 0, 1], 'o')
#                plt.axis('equal')
#                break
            
        #%%
        skeletons_sorted = np.concatenate(skeletons_sorted)
        trajectories_data = pd.concat(trajectories_data)
        
        #%%
        with tables.File(str(skel_file_nn), 'w') as fid:
        
            fid.create_table('/',
                        "trajectories_data",
                        obj = trajectories_data.to_records(index=False),
                        filters = TABLE_FILTERS)
            
            fid.create_carray('/', 
                            'skeleton',
                            obj = skeletons_sorted.astype(np.float32),
                            filters = TABLE_FILTERS
                            )
        
        #%%
        
    #%%
#    for frame, frame_data in trajectories_data.groupby('frame_number'):
#        skel_ids = frame_data['skeleton_id'].values
#        skels = skeletons_sorted[skel_ids]
#        
#        for ss in skels:
#            plt.plot(ss[:, 0], ss[:, 1], '.-')
        
        #%%
        #mid = skeletons.shape[1]//2
        
#    #%%
#    
#    with tables.File(str(mask_file), 'r') as fid:
#        masks = fid.get_node('/mask')
#        for frame, frame_data in skel_info.groupby('frame_number'):
#            frame_skeletons = skeletons[frame_data['skeleton_id']]
#            
#            img = masks[frame]
#            #%%
#            plt.figure(figsize = (15, 15))
#            plt.imshow(img, cmap='gray')
#            for ss in frame_skeletons:
#                plt.plot(ss[:, 0], ss[:, 1], '.-')
#                plt.plot(ss[mid, 0], ss[mid, 1], 'o')
            
        
#    plt.figure(figsize = (15, 15))
#    #plt.imshow(img, cmap='gray')
#    for ss in worm_skels[2150:2170]:
#        plt.plot(ss[:, 0], ss[:, 1], 'k.-')
#        plt.plot(ss[0, 0], ss[0, 1], 'or')
        