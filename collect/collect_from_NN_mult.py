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
import tqdm

if __name__ == '__main__':
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/mating/wildMating3.2_MY23_cross_MY23_cross_PC1_Ch1_17082018_123407.hdf5'
    
    #skels_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/Results/JU792_Ch1_24092017_063115_skeletonsNN.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/MaskedVideos/JU792_Ch1_24092017_063115.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/unc-104 (e1265)III on food L_2011_10_18__13_23_55___1___7.hdf5'
    mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Syngenta/N2_worms10_CSAA026102_100_Set7_Pos4_Ch3_14072017_195800.hdf5'
    feats_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
    skels_file = mask_file.replace('.hdf5', '_skeletonsNN.hdf5')
    
    with pd.HDFStore(feats_file, 'r') as fid:
        clusters_data = fid['/trajectories_data']
    
    with pd.HDFStore(skels_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        trajectories_data = trajectories_data
    
    clusters_frame_g = clusters_data.groupby('frame_number').groups
     
    clusters_frame_g = clusters_data.groupby('frame_number')
    traj_frame_g = trajectories_data.groupby('frame_number')
    
    tot = clusters_data['frame_number'].max()
    #%%
    cluster_sizes = {}
    cluster_matches = {}
    
    for (frame, frame_data), (_, frame_clusters) in tqdm.tqdm(zip(traj_frame_g, clusters_frame_g), total = tot):
        
        for _, row_cluster in frame_clusters.iterrows():
            
            rec = frame_data.to_records()
            dx = rec['coord_x'] - row_cluster['coord_x']
            dy = rec['coord_y'] - row_cluster['coord_y']
            rr = np.sqrt((dx*dx + dy*dy))
            r_lim =  row_cluster['roi_size']//2
            
            _matches = frame_data.index.values[rr < r_lim]
            cluster_size = len(_matches)
            if cluster_size:
            
                w_ind = int(row_cluster['worm_index_joined'])
                if not w_ind in cluster_sizes:
                    cluster_sizes[w_ind] = (frame, cluster_size, _matches)
                else:
                    prev_size = cluster_sizes[w_ind][1]
                    if cluster_size >= prev_size:
                        cluster_sizes[w_ind] = (frame, cluster_size, _matches)
                
                for ind in _matches:
                    cluster_matches[ind] = w_ind
        
    #%%
        
    rois2plot = []
    traj_g = trajectories_data.groupby('worm_index_joined').groups
    traj_g = [x for x in traj_g.items() if x[1].size > 500]
    traj_g = sorted(traj_g, key = lambda x : len(x[1]))
    
    for w_ind, inds in traj_g[::-1]:
        worm_data = trajectories_data.loc[inds]
        
        frames = worm_data['timestamp_raw'].values
        frames2check, = np.where(np.diff(frames) > 1)
        #frames2check = frames2check.tolist() 
        #frames2check.append(len(worm_data)-1)
          
        roi_candidates = worm_data.iloc[frames2check]
                
        for roi_i, roi_data in roi_candidates.iterrows():
            if roi_i in cluster_matches:
                cluster_id = cluster_matches[roi_i]
                cluster_size = cluster_sizes[cluster_id][1]
                if (cluster_size == 1):
                    rois2plot.append(roi_data)
           
            
    rois2plot = pd.DataFrame(rois2plot)
    #%%
    
    with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
        masks = fid_mask.get_node('/mask')
        skeletons = fid_skel.get_node('/coordinates/skeletons')
        
        for frame, frame_data in rois2plot.groupby('timestamp_raw'):
            
            img = masks[int(frame)]
            
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
                
                
                plt.title((frame, row['worm_index_joined']))
            
    #%%
    
    real_clusters = {}
    for frame, cluster_size, dat in cluster_sizes.values():
        if cluster_size > 1:
            if not frame in real_clusters:
                real_clusters[frame] = []
            real_clusters[frame].append(dat)
        
    _offset = 5
    with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
        masks = fid_mask.get_node('/mask')
        skeletons = fid_skel.get_node('/coordinates/skeletons')
        
        tot, H, W = masks.shape
        
        skeletons = fid_skel.get_node('/coordinates/skeletons')
        for frame, data in real_clusters.items():
            img = masks[int(frame)]
            
            for roi_ind in data:
                roi_data = trajectories_data.loc[roi_ind].to_records()
                
                skel_ids = roi_data['skeleton_id']
                skels = skeletons[skel_ids, :, :]
                
                roi_offset = roi_data['roi_size']//2 + _offset
                xl = roi_data['coord_x'] - roi_offset
                xl = int(max(0, xl.min()))
                xr = roi_data['coord_x'] + roi_offset
                xr = int(xr.max())
                
                yl = roi_data['coord_y'] - roi_offset
                yl = int(max(0, yl.min()))
                
                yr = roi_data['coord_y'] + roi_offset
                yr = int(yr.max())
                
                roi = img[yl:yr, xl:xr]
                
                
                skels[..., 0] -= xl
                skels[..., 1] -= yl
                
                plt.figure()
                plt.imshow(roi, cmap = 'gray')
                for skel in skels:
                    plt.plot(skel[:, 0], skel[:, 1])
                
            
            
                
                
                
                
    
    
    
    
    
    
    
    