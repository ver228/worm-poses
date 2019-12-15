#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:43:39 2019

@author: avelinojaver
"""
import tables
import numpy as np
from pathlib import Path
import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
#%%
def _get_best_match(_edges, min_PAF = 0.2):
    _edges = _edges[_edges['cost_PAF'] >= min_PAF]
    _edges = _edges[np.argsort(_edges['cost_PAF'])[::-1]] #sort with the largest PAF first...
    _, valid_index =  np.unique(_edges['point1'], return_index = True ) #unique returns the first occurence (the largest value on the sorted list)
    return _edges[valid_index]
#%%
if __name__ == '__main__':
    
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/unlinked/')
    masked_file = root_dir / 'wildMating3.1_MY23_self_MY23_self_PC1_Ch1_17082018_121653.hdf5'
    unlinked_file = root_dir / (masked_file.stem + '_unlinked-skels.hdf5')
    
    
    with tables.File(unlinked_file) as fid:
        points = fid.get_node('/points')[:]
        edges = fid.get_node('/edges')[:]
    
    
    #%%
    #grouping points...
    n_segments = 8
    tot_frames = points['frame_number'].max() + 1
    points_g = [[[] for _ in range(n_segments)] for _ in range(tot_frames)]
    for irow, row in enumerate(points):
        points_g[row['frame_number']][row['segment_id']].append(irow)
    
    edges_g = [[] for _ in range(tot_frames)]
    for irow, row in enumerate(edges):
        edges_g[row['frame_number']].append(irow)
    #%%
    
    skeletons = []
    missing_points = []
    missing_halves = []
    for frame in tqdm.trange(tot_frames):
        
        edges_in_frame = edges[edges_g[frame]]
        best_matches = _get_best_match(edges_in_frame)
        best_matches = {x['point1']:x['point2'] for x in best_matches}
        
        
        skels_linked = []
        
        matched_points = set(best_matches.keys())
        
        
        points_in_frame = points[[x for c in points_g[frame] for x in c]]
        points_in_frame_g = {x['point_id']:x for x in points_in_frame}
        
        for seg_id in range(n_segments):
            points_in_segment = points[points_g[frame][seg_id]]
            
            prev_inds = {x[-1]['point_id']:x for x in skels_linked}
            matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
            matched_indices_cur = [best_matches[x] for x in matched_indices_prev]
            
            new_points = set(points_in_segment['point_id']) - set(matched_indices_cur)
            
            
            for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
                prev_inds[k1].append(points_in_frame_g[k2])
            skels_linked += [[points_in_frame_g[x]] for x in new_points]
    
        matched_halves = defaultdict(list)
        bad_index = []
        for _half in skels_linked:
            _half = np.array(_half)
            if len(_half) == n_segments:
                midbody_ind = _half[-1]['point_id']
                matched_halves[midbody_ind].append(_half)
            else:
                bad_index += _half['point_id'].tolist()
        
        skeletons_in_frames = []
        missing_halves_in_frame = []
        for k, _halves in matched_halves.items():
            _halves = [np.array((h['x'],h['y'])).T for h in _halves]
            if len(_halves) == 2:
                h1, h2 = _halves
                skel = np.concatenate((h1, h2[-2::-1]))
                
                skeletons_in_frames.append(skel)
            else:
                missing_halves_in_frame += _halves
                #for h in _halves:
                #    bad_index += h['point_id'].tolist()
        
        skeletons.append(skeletons_in_frames)
        missing_halves.append(missing_halves_in_frame)
        
        missing_in_frame = np.array([points_in_frame_g[p] for p in bad_index])
        missing_points.append(missing_in_frame)
    #%% #link on time
    def _skel2halfs(_skels, n_segments = 8):
        return np.concatenate((_skels[:, :n_segments], _skels[:, (n_segments-1):][:, ::-1]))
        
    max_frac_dist = 0.5
    
    current_id = 0
    prev_data = None
    
    dat = zip(skeletons, missing_halves, missing_points)
    for frame, (skels, halves, missing) in enumerate(tqdm.tqdm(dat, total = len(skeletons))):
        skels = np.array(skels)
        
        if prev_data is None:
            skel_ids = list(range(len(skels)))
            prev_data = [skel_ids, skels]
            
            matched_skeletons = [[(frame, s)] for s in skels]
            current_id += len(matched_skeletons)
        else:
            prev_skel_id, prev_skels = prev_data
            prev_length = np.sqrt(np.diff(prev_skels, axis=1)**2).sum(-1).sum(-1)
            
            prev_half = _skel2halfs(prev_skels)
            curr_half = _skel2halfs(skels)
            
            rr = prev_half[:, None] - curr_half[None, ...]
            
            cost = np.sqrt((rr**2).sum(-1)).sum(-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            match_cost = cost[row_ind, col_ind]
            
            L = np.concatenate((prev_length, prev_length))
            cost_limits = (max_frac_dist*L[row_ind])
            is_valid_cost = match_cost < cost_limits
            row_ind, col_ind = row_ind[is_valid_cost], col_ind[is_valid_cost]
            
            n_skels = len(skels)
            prev_ids = row_ind % len(prev_skels)
            prev_half_id = row_ind // len(prev_skels)
            
            next_ids = col_ind % n_skels
            missing_skels = list(set(range(n_skels)) - set(next_ids))
            
            dat2match = {k : [-1, -1] for k in prev_ids}
            for pid, hid, nid in zip(prev_ids, prev_half_id, col_ind):
                dat2match[pid][hid] = nid
            
            dat2match = {k:v for k,v in dat2match.items() if v[0]%n_skels ==  v[1]%n_skels}
            
            next_skels = []
            next_skel_ids = []
            for pid, (nid1, nid2) in dat2match.items():
                skel_n = np.concatenate((curr_half[nid1], curr_half[nid2][-2::-1]))
                
                
                pid_skel = prev_skel_id[pid]
                matched_skeletons[pid_skel].append((frame, skel_n))
                
                next_skels.append(skel_n)
                next_skel_ids.append(prev_skel_id[pid])
            if not next_skels:
                next_skels = np.zeros((0, skels.shape[0]))
            
            
            n2add = len(missing_skels)
            skelids2add =  [i+ current_id for i in range(n2add)]
            current_id += n2add
            
            skels2add = skels[missing_skels]
            
            next_skel_ids +=  skelids2add
            next_skels = np.concatenate((next_skels, skels2add))
            
            matched_skeletons += [[(frame, s)] for s in skels2add]
            
            prev_data = [next_skel_ids, next_skels]
            
    matched_skeletons = [x for x in matched_skeletons if len(x) > 1]
    #%%
    TABLE_FILTERS = tables.Filters(
                        complevel=5,
                        complib='zlib',
                        shuffle=True,
                        fletcher32=True)


    
    tables_dtypes = np.dtype([
        ("timestamp_raw", np.int32),
        ("timestamp_time", np.float32),
        ("worm_index_joined", np.int32),
        ("coord_x", np.float32),
        ("coord_y", np.float32),
        ("roi_size", np.float32),
        ("threshold", np.float32),
        ("area", np.float32),
        ("frame_number", np.int32),
        ("was_skeletonized", np.uint8),
        ("skeleton_id", np.int32)
        ])
    
    # save results
    skel_file = unlinked_file = root_dir / (masked_file.stem + 'skelsNN.hdf5')
    
    
    curr_skel_id = 0
    tot_skeletons = sum([len(x) for x in matched_skeletons])
    with tables.File(str(skel_file), 'w') as fid:
        
        
        inds_tab = fid.create_table('/',
                    "trajectories_data",
                    tables_dtypes,
                    filters = TABLE_FILTERS
                    )
        
        w_node = fid.create_group('/', 'coordinates')
        skel_array = fid.create_carray(w_node, 
                        'skeletons',
                        atom = tables.Float32Atom(),
                        shape = (tot_skeletons, 15, 2),
                        chunkshape = (1, 15, 2),
                        filters = TABLE_FILTERS
                        )
        
        for ii, worm_data in enumerate(tqdm.tqdm(matched_skeletons)):
            
            _, skels = zip(*worm_data)
            
            skels = np.array(skels)
            
            
            
            coord_min = skels.min(axis=1)
            coord_max = skels.max(axis=1)
            roi_size = (coord_max - coord_min).max(axis=1).max()
            cms = (coord_max + coord_min)/2
            skel_ids = np.arange(curr_skel_id, curr_skel_id + len(skels))
            curr_skel_id += len(skels)
            
            
            tabs_data = []
            worm_id = ii + 1
            for (frame,  skel), (cx, cy), skel_id in zip(worm_data, cms, skel_ids):
                row = (frame, np.nan, worm_id, cx, cy, roi_size, 0., 0., frame, 1, skel_id)            
                tabs_data.append(row)
            
            
            skel_array[skel_ids[0]:skel_ids[-1]+1] = skels.astype(np.float32)
            inds_tab.append(tabs_data)
            