#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:11:35 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.models import CPM, CPM_PAF
#%%
import torch
import tables
import numpy as np

from pathlib import Path

import math
import pandas as pd
import cv2
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment




def _get_device(cuda_id = 0):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device
             
#%%
def cv2_peak_local_max(img, threshold_relative, threshold_abs):
    #max_val = img.max()
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    th = max(max_val*threshold_relative, threshold_abs)
    
    _, mm = cv2.threshold(img, th, max_val, cv2.THRESH_TOZERO)
    #max filter
    kernel = np.ones((3,3))
    mm_d = cv2.dilate(mm, kernel)
    loc_maxima = cv2.compare(mm, mm_d, cv2.CMP_GE)
    
    mm_e = cv2.erode(mm, kernel)
    non_plateau = cv2.compare(mm, mm_e, cv2.CMP_GT)
    loc_maxima = cv2.bitwise_and(loc_maxima, non_plateau)
    
    _, coords, _ = cv2.findContours(loc_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coords = np.array([x.squeeze()[::-1] for cc in coords for x in cc])
    coords = np.array(coords)
    #coords = np.array(np.where(loc_maxima>0)).T
    #the code above is faster than  coords = np.array(np.where(loc_maxima>0)).T
    return coords

def _get_peaks(cpm_maps, threshold_relative, threshold_abs):
    all_coords = []
    for mm in cpm_maps:
        coords = cv2_peak_local_max(mm, threshold_relative, threshold_abs)
        all_coords.append(coords)
        
        
    return all_coords  


#%%
def _get_edges(points_coords, paf_maps, paf_map_dist = 5, max_edge_dist = 50, _is_debug = False):
    
    edges_vals = []
    for t1, paf_map in enumerate(paf_maps):
        
        t2 = t1 + paf_map_dist
        points1 = points_coords[t1]
        points2 = points_coords[t2]
        
        if _is_debug:
            import matplotlib.pylab as plt
            plt.figure()
            plt.imshow(paf_map[0])
            plt.plot(points1[..., 1], points1[..., 0], 'vr')
            plt.plot(points2[..., 1], points2[..., 0], 'or')
        
        if len(points1) == 0 or len(points2) == 0:
            continue
        
        M = np.zeros((len(points1), len(points2)), np.float32)
        D = np.zeros_like(M)
        for i1, v1 in enumerate(points1):
            for i2, v2 in enumerate(points2):
                _shift = v2 - v1
                _mag = np.linalg.norm(_shift)
                
                if _mag > max_edge_dist or _mag == 0:
                    val = 0
                else:
                    u = _shift / _mag
                    
                    if u[0] != 0 and u[1] != 0:
                        ci = np.arange(0, _shift[0], u[0])
                        cj = np.arange(0, _shift[1], u[1])
                    
                    elif u[0] == 0:
                        cj = np.arange(0, _shift[1], u[1])
                        ci = np.zeros_like(cj)
                    else:
                        ci = np.arange(0, _shift[0], u[0])
                        cj = np.zeros_like(ci)
                    
                    assert len(ci) == len(cj)
                    
                    ci = ci.round().astype(np.int) + v1[0]
                    cj = cj.round().astype(np.int) + v1[1]
                    
                    L = paf_map[:, ci, cj]
                    #the coordinates are inverted too...
                    u_inv = u[::-1]
                    val = np.mean(np.dot(u_inv[None], L))
                    
                M[i1, i2] = max(0, val)
                D[i1, i2] = _mag
        
        cost_matrix = M.copy()
        if M.shape[0] < M.shape[1]:
            cost_matrix = cost_matrix.T
        
        #I am adding a second column that cost the double just to allow for cases 
        #where it might be convenient to add to the same edge
        cost_matrix = np.concatenate((cost_matrix, cost_matrix/2), axis=1)
        
        #convert maximization to a minimization problem
        cost_matrix = cost_matrix.max() - cost_matrix
        col_ind, row_ind = linear_sum_assignment(cost_matrix)
        
        
        if M.shape[0] < M.shape[1]:
            row_ind, col_ind = col_ind, row_ind
        
        row_ind = row_ind % M.shape[1]
        col_ind = col_ind % M.shape[0]
        
        for ind1, ind2 in zip(col_ind, row_ind):
            if D[ind1, ind2] > max_edge_dist:
                continue
            
            row = (t1, ind1, t2, ind2, M[ind1, ind2], D[ind1, ind2])
            edges_vals.append(row)
            
            if _is_debug:
                y, x = zip(points1[ind1], points2[ind2])
                
                plt.plot(x,y, 'r')
            
        #if t1 > 0: break
        
    edges_vals = pd.DataFrame(edges_vals, columns = ['map1', 'ind1', 'map2', 'ind2', 'val', 'size'])  
    
    return edges_vals
#%%
#points_coords, paf_maps, paf_map_dist, max_edge_dist, _is_debug = all_coords, paf_maps_inv, 5, 20, True 
#%%
def _link_skeletons(points_coords, paf_maps, paf_map_dist = 5, max_edge_dist = 20, _is_debug = False):
    #%%
    max_edge_size = paf_map_dist*max_edge_dist
    edges_vals_df = _get_edges(points_coords, 
                               paf_maps, 
                               paf_map_dist = paf_map_dist,
                               _is_debug = _is_debug)
    edges_vals_df = edges_vals_df[edges_vals_df['size']<=max_edge_size]
    
    if len(edges_vals_df) == 0:
        return []
    #%%
    #I making the strong assumtion that most of the maps will have identified a correct number of edges
    expected_n_halfs = math.floor(edges_vals_df['map1'].value_counts().median())
    
    edges_g = edges_vals_df.groupby('map1')
    all_edges = []
    for map1, dat in  edges_g:
        edges = []
        for irow, row in dat.iterrows():
            p1 = points_coords[int(row[0])][int(row[1])]
            p2 = points_coords[int(row[2])][int(row[3])]
            edges.append(np.array((p1, p2)))
        all_edges.append(edges)
    
    #%%
    
    for ini_ind, edges in  enumerate(all_edges):
        if len(edges) == expected_n_halfs:
            seeds = [[x] for x in edges]
    
    ini_ind = 0
    seeds = [[x] for x in all_edges[0]]
    
    def _get_cost(seeds, edges, seed_ind = -1):
        
        cost_next = []
        for s in seeds:
            c = []
            for e in edges:
                dd = (s[seed_ind] - e)
                val = np.linalg.norm(dd, axis=1).sum(0)
                c.append(val)
            cost_next.append(c)
        cost_next = np.array(cost_next)
        
        
        cost_same = np.full((len(seeds), len(edges)), max_edge_dist, np.float32)
        
        cost_matrix = np.concatenate((cost_next, cost_same), axis=1)
        
        return cost_matrix
    
    for edges in all_edges[ini_ind+1:]:
        cost_matrix = _get_cost(seeds, edges, seed_ind = -1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if c < len(edges) and cost_matrix[r, c] < max_edge_dist:
                seeds[r].append(edges[c])
            
        missing_edges = set(range(len(edges))) - set(col_ind)
        for c in missing_edges:
            seeds.append([edges[c]])
    
    skel_halfs = []
    for ss in seeds:
        half_ = [x[0] for x in ss] + [x[1] for x in ss[-paf_map_dist:]]
        skel_halfs.append(half_)
    skel_halfs = [x for x in skel_halfs if len(x) > 20]
    if _is_debug:
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
        for ss in skel_halfs:
            ss = np.array(ss)
            plt.plot(ss[:, 1], ss[:, 0])
    
    #%%
    #Now let's link the halfs
    #TODO Maybe I should introduce an extra affinity map for this...
    
    skel_mid = [x[-1] for x in skel_halfs]
    skel_mid_next = [np.mean(x[-5:-1], axis=0) for x in skel_halfs]
    
    pairs = list(map(np.array, zip(skel_mid, skel_mid_next)))
    
    C = np.full((len(pairs), len(pairs)), 300.0, np.float32)
    for i1, p1 in enumerate(pairs):
        for i2, p2 in enumerate(pairs):
            
            if i1 != i2:
                #let's introduce two cost
                #Remember, last index is the midbody
                
                #the first is aligment midline segments should go to different directions
                #so their addition of the directional vector should be zero. 
                #This helps to link halfs from the worms aggregates. 
                v1 = p1[-2] - p1[-1]
                v2 = p2[-2] - p2[-1]
                cost_aligment = np.linalg.norm(v1+v2).sum(0)
                
                #The second is the distance, but I will only would like to introduce this if 
                #the segments are too far away, so let's do it a step function using max_edge_dist
                _dist = np.linalg.norm((p1[-1] - p2[-1]))
                cost_dist = _dist if _dist >= max_edge_dist else 0
                
                cost = cost_dist + cost_aligment
                C[i1, i2] = cost
            
    row_ind, col_ind = linear_sum_assignment(C)
    
    #matches = [(r,c) if c > r else (c,r) for r,c in zip(row_ind, col_ind) if r != c]
    matches = [(r,c) if c > r else (c,r) for r,c in zip(row_ind, col_ind)]
    matches = set(matches)
    #assert len(matches) == expected_n_worms
    
    skels = []
    for i1, i2 in matches:
        #p1, p2 = map(np.array, (skel_halfs[i1], skel_halfs[i2]))
        #print(i1, i2, np.linalg.norm((p1[-1] - p2[-1])))
        skel = np.concatenate((skel_halfs[i1], skel_halfs[i2][-1::-1]))
        skels.append(skel)
    
    if _is_debug:
        plt.figure()
        plt.imshow(roi, cmap='gray')
        for ss in skels:
            plt.plot(ss[:, 1], ss[:, 0])
    
    return skels
#%%
def maps2skels(cpm_maps, paf_maps):
    #%%
    all_coords = _get_peaks(cpm_maps, threshold_relative = 0.5, threshold_abs = 0.05)
    
    
    
    paf_maps_inv = paf_maps[::-1]
    skels = _link_skeletons(all_coords, paf_maps_inv, paf_map_dist = 5)
    #%%
    return skels
    
#%%
if __name__ == '__main__':
    from tierpsy.analysis.ske_create.helperIterROI import getROIfromInd
    #_model_path = Path.home() / 'workspace/WormData/results/worm-poses/logs/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'
    
    #_model_path = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/trained_models/manually-annotated_20190115_140602_CPMout_adam_lr0.0001_wd0.0_batch24/checkpoint.pth.tar'
    #n_segments = 25
    #model = CPM(n_segments = n_segments, 
    #                         same_output_size = True)
    
    _model_path = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/trained_models/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'
    n_segments = 25
    n_affinity_maps = 20
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True)
   
    
    cuda_id = 3
    device = _get_device(cuda_id)
    
    
    state = torch.load(_model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    model = model.to(device)
    model.eval()
    #%%
    
    mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5'
    pairs = [(7,68), (2, 654), (1866, 11100), (2106, 13464), (4027, 22150)]
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
    #pairs = [(3164, 12732), (5155, 22150)]
            
    
    feat_file = mask_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_featuresN.hdf5')
    
    with pd.HDFStore(str(feat_file), 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    if True:
        for worm_index, frame_number in pairs:
            row_data, roi, roi_corner = getROIfromInd(mask_file, trajectories_data, frame_number, worm_index, roi_size=-1)
            
            with torch.no_grad():
                X = torch.tensor(roi[None, None]).float()
                X = X.to(device)
                X /= 255.
                outs = model(X)
            
            cpm_maps, paf_maps = outs[-1]
            paf_maps = paf_maps[0].detach().cpu().numpy()
            cpm_maps = cpm_maps[0].detach().cpu().numpy()
            
            paf_maps_inv = paf_maps[::-1]
            
            
            all_coords = _get_peaks(cpm_maps, threshold_relative = 0.25, threshold_abs = 0.05)
            skels = _link_skeletons(all_coords, paf_maps_inv, paf_map_dist = 5)
            
            
            plt.figure()
            plt.imshow(roi, cmap='gray')
            for ss in skels:
                plt.plot(ss[:, 1], ss[:, 0])
    
    
    if False:
        with tables.File(str(mask_file)) as fid:
            frame_number =  22150 #12732#11100#
            roi = fid.get_node('/mask')[frame_number]
            
        with torch.no_grad():
            X = torch.tensor(roi[None, None]).float()
            X = X.to(device)
            X /= 255.
            outs = model(X)
        
        cpm_maps, paf_maps = outs[-1]
        paf_maps = paf_maps[0].detach().cpu().numpy()
        cpm_maps = cpm_maps[0].detach().cpu().numpy()
        
        paf_maps_inv = paf_maps[::-1]
        
        #%%
        all_coords = _get_peaks(cpm_maps, threshold_relative = 0.5, threshold_abs = 0.05)
        #%%
        plt.figure()
        plt.imshow(roi, cmap='gray')
        ss = np.concatenate(all_coords)
        plt.plot(ss[:, 1], ss[:, 0], '.r')
        #%%
        
        #%%
        skels = _link_skeletons(all_coords, paf_maps_inv, paf_map_dist = 5)
        #%%
        plt.figure()
        plt.imshow(roi, cmap='gray')
        for ss in skels:
            plt.plot(ss[:, 1], ss[:, 0])
            plt.plot(ss[0, 1], ss[0, 0], 'o')