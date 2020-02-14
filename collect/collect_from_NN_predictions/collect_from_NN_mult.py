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
from pathlib import Path
import pickle

def get_hard_rois(mask_file, feats_file, skels_file, min_traj_size = 500):
    with pd.HDFStore(feats_file, 'r') as fid:
        clusters_data = fid['/trajectories_data']
    
    with pd.HDFStore(skels_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        trajectories_data = trajectories_data
    
    clusters_data_rec = clusters_data.to_records()
    trajectories_data_rec = trajectories_data.to_records()
    
    tot_frames = trajectories_data_rec['frame_number'].max()
    
    
    def groupby_frame(rec_array, col = 'frame_number'):
        tot_frames = max(rec_array[col])
        grouped_data = [[] for _ in range(tot_frames + 1)]
        
        for irow, row in enumerate(rec_array):
            grouped_data[row[col]].append(irow)
        return grouped_data
    
    clusters_g = groupby_frame(clusters_data_rec)
    traj_g = groupby_frame(trajectories_data_rec)
    
    
    
    
    cluster_sizes = {}
    cluster_matches = {}
    
    for frame in tqdm.trange(tot_frames):
        
        frame_cluster_ids = clusters_g[frame]
        if not len(frame_cluster_ids):
            continue
        frame_clusters = clusters_data_rec[frame_cluster_ids]
        
        frame_data_inds = np.array(traj_g[frame])
        if not len(frame_data_inds):
            continue
        
        frame_data = trajectories_data_rec[frame_data_inds]
        for row_cluster in frame_clusters:
            dx = frame_data['coord_x'] - row_cluster['coord_x']
            dy = frame_data['coord_y'] - row_cluster['coord_y']
            rr = np.sqrt((dx*dx + dy*dy))
            r_lim =  row_cluster['roi_size']//2
            
            _matches = frame_data_inds[rr < r_lim]
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
        
        
     
    
    
    traj_g = trajectories_data.groupby('worm_index_joined').groups
    traj_g = [x for x in traj_g.items() if x[1].size > min_traj_size]
    traj_g = sorted(traj_g, key = lambda x : len(x[1]))
    
    rois2check = []
    for w_ind, inds in traj_g[::-1]:
        worm_data = trajectories_data.loc[inds]
        
        frames = worm_data['frame_number'].values
        frames2check, = np.where(np.diff(frames) > 1)
        #frames2check = frames2check.tolist() 
        #frames2check.append(len(worm_data)-1)
          
        roi_candidates = worm_data.iloc[frames2check]
                
        for roi_i, roi_data in roi_candidates.iterrows():
            if roi_i in cluster_matches:
                cluster_id = cluster_matches[roi_i]
                cluster_size = cluster_sizes[cluster_id][1]
                if (cluster_size == 1):
                    rois2check.append(roi_data)
           
            
    rois2check = pd.DataFrame(rois2check)
    
    
    
    all_ROIs_coils = []
    
    if len(rois2check):
        with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
            masks = fid_mask.get_node('/mask')
            skeletons = fid_skel.get_node('/coordinates/skeletons')
            
            for frame, frame_data in rois2check.groupby('frame_number'):
                
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
                    
                    all_ROIs_coils.append((roi, skel[None], (xl, yl), frame))
                
    
    def _get_cross_score(ss):
        
        L = np.linalg.norm(np.diff(ss, axis=0), axis=1).sum()
        ss = ss[::8]
        
        rr = np.linalg.norm((ss[None] - ss[:, None]), axis=-1)
        np.fill_diagonal(rr, 1e10)
        
        rr[rr < 5] = 10
        
        return rr.min()/L
    
    all_ROIs_coils = sorted(all_ROIs_coils, key = lambda x : _get_cross_score(x[1][0]))
    
            
    
    real_clusters = {}
    for frame, cluster_size, dat in cluster_sizes.values():
        if cluster_size > 1:
            if not frame in real_clusters:
                real_clusters[frame] = []
            real_clusters[frame].append(dat)
        
    
    all_ROIs_clusters = []
    
    if len(real_clusters):
        with tables.File(skels_file, 'r') as fid_skel, tables.File(mask_file, 'r') as fid_mask:
            masks = fid_mask.get_node('/mask')
            skeletons = fid_skel.get_node('/coordinates/skeletons')
            
            tot_frames, H, W = masks.shape
            
            skeletons = fid_skel.get_node('/coordinates/skeletons')
            for frame, data in real_clusters.items():
                img = masks[int(frame)]
                
                for roi_ind in data:
                    roi_data = trajectories_data.loc[roi_ind].to_records()
                    
                    skel_ids = roi_data['skeleton_id']
                    skels = skeletons[skel_ids, :, :]
                    
                    roi_offset = roi_data['roi_size']//2 #+ _offset
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
                    
                    all_ROIs_clusters.append((roi, skels, (xl, yl), frame))
    
    def _get_cluster_score(ss):
        rr = np.linalg.norm((ss[None] - ss[:, None]), axis=-1)
        rr = np.percentile(rr, 10, axis=-1)
        np.fill_diagonal(rr, 1e10)
        score = rr.min()/len(ss)
        
        return score
    
    all_ROIs_clusters = sorted(all_ROIs_clusters, key = lambda x : _get_cluster_score(x[1]))
    
    return all_ROIs_coils, all_ROIs_clusters, tot_frames


def _test():
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/mating/wildMating3.2_MY23_cross_MY23_cross_PC1_Ch1_17082018_123407.hdf5'
    mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Syngenta/N2_worms10_CSAA026102_100_Set7_Pos4_Ch3_14072017_195800.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Serena_WT_Screening/15.1_5_cx11314_da_cb4852_ff_Set0_Pos0_Ch1_03032018_145825.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Pratheeban/X3_earlyL1_2.0_Ch1_20072016_161147.hdf5'
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Pratheeban/03_L3_1.6_Ch1_24062016_153519.hdf5'
    
    
    
    feats_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
    skels_file = mask_file.replace('.hdf5', '_skeletonsNN.hdf5')
    all_ROIs_coils, all_ROIs_clusters = get_hard_rois(mask_file, feats_file, skels_file)
    
    for roi, skels, corner, frame in all_ROIs_coils[:max_n_coils]:
        plt.figure()
        for skel in skels:
            plt.imshow(roi, cmap = 'gray')
            plt.plot(skel[:,0], skel[:,1])
        
    
    for roi, skels, corner, frame in all_ROIs_clusters[:max_n_clusters]:
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
        
        
        for skel in skels:
            plt.plot(skel[:, 0], skel[:, 1])
            
#%%

if __name__ == '__main__':
    save_dir_root = Path.home() / 'workspace/WormData/worm-poses/rois2select/v1'
    
    
    screens_dir = Path.home() / 'workspace/WormData/screenings/'
    subdirs2check  = [
                      ('Serena_WT_Screening', 150), 
                      ('Pratheeban',  100),
                      ('mating_videos/Mating_Assay', 50),
                      ('mating_videos/wildMating', 50),
                      ('pesticides_adam/Syngenta', 25)
                      ]
    
    
    for subdir, max_n_rois in subdirs2check:
        save_dir = save_dir_root / subdir 
        save_dir.mkdir(parents = True, exist_ok = True)
        
        root_dir = screens_dir / subdir
        f_subdirs = dict( feats = 'Results', masks = 'MaskedVideos', skelsNN = 'ResultsNN_v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32')
        postfixes = dict( feats = '_featuresN.hdf5', masks = '.hdf5', skelsNN = '_skeletonsNN.hdf5')
        
        skel_files =  [x for x in (root_dir / f_subdirs['skelsNN']).rglob('*' + postfixes['skelsNN'])]
        
        for skels_file in tqdm.tqdm(skel_files, desc = f'{subdir} | Processing...'):
            
            mask_file = Path(str(skels_file).replace(f_subdirs['skelsNN'], f_subdirs['masks']).replace(postfixes['skelsNN'], postfixes['masks']))
            feats_file = Path(str(skels_file).replace(f_subdirs['skelsNN'], f_subdirs['feats']).replace(postfixes['skelsNN'], postfixes['feats']))
            
            
            save_name = save_dir / (mask_file.stem + '_ROIs.p')
            
            if save_name.exists():
                continue
            
            assert mask_file.exists() and skels_file.exists()
            
            all_ROIs_coils, all_ROIs_clusters, tot_frames = get_hard_rois(mask_file, feats_file, skels_file)
            
            all_ROIs_coils = all_ROIs_coils[:max_n_rois]
            all_ROIs_clusters = all_ROIs_clusters[:max_n_rois]
            
            
            with open(save_name, 'wb') as fid:
                rois_d = {'coils':all_ROIs_coils, 'clusters':all_ROIs_clusters}
                pickle.dump(rois_d, fid)
            
            
            
    #%%
                
    for roi, skels, corner, frame in dat['clusters'][50:100]:
        fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
        for ax in axs:
            ax.imshow(roi, cmap = 'gray')
            
        inds = np.arange(0, skels.shape[1] + 1, 4)
        for skel in skels:
            plt.plot(skel[inds, 0], skel[inds, 1], '.-')
        
        