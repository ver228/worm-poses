#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:10:35 2020

@author: avelinojaver
"""
import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/mating/wildMating3.2_MY23_cross_MY23_cross_PC1_Ch1_17082018_123407.hdf5'
    
    #skels_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/Results/JU792_Ch1_24092017_063115_skeletonsNN.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/MaskedVideos/JU792_Ch1_24092017_063115.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/unc-104 (e1265)III on food L_2011_10_18__13_23_55___1___7.hdf5'
    mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Syngenta/N2_worms10_CSAA026102_100_Set7_Pos4_Ch3_14072017_195800.hdf5'
    
    skels_file = mask_file.replace('.hdf5', '_skeletonsNN.hdf5')
    
    
    rois2plot = []
    with pd.HDFStore(skels_file, 'r') as fid:
        
        trajectories_data = fid['/trajectories_data']
        traj_g = trajectories_data.groupby('worm_index_joined').groups
        traj_g = [x for x in traj_g.items() if x[1].size > 500]
        traj_g = sorted(traj_g, key = lambda x : len(x[1]))
        
        for w_ind, inds in traj_g[::-1]:
            worm_data = trajectories_data.loc[inds]
            
            frames = worm_data['timestamp_raw'].values
            frames2check, = np.where(np.diff(frames) > 1)
            rois2plot.append(worm_data.iloc[frames2check])
            
    rois2plot = pd.concat(rois2plot)
    #%%
    is_stage_moving = None
    with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
        masks = fid_mask.get_node('/mask')
        skeletons = fid_skel.get_node('/coordinates/skeletons')
        
        if '/stage_position_pix' in fid_mask:
            stage_position_pix = fid_mask.get_node('/stage_position_pix')[:]
            is_stage_moving = np.isnan(stage_position_pix)
        
        for frame, frame_data in rois2plot.groupby('timestamp_raw'):
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
                
                plt.figure()
                plt.imshow(roi, cmap = 'gray')
                plt.plot(skel[:,0], skel[:,1])
                
                