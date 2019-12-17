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
def _get_best_edge_match(_edges, min_PAF = 0.2):
    _edges = _edges[_edges['cost_PAF'] >= min_PAF]
    _edges = _edges[np.argsort(_edges['cost_PAF'])[::-1]] #sort with the largest PAF first...
    _, valid_index =  np.unique(_edges['point1'], return_index = True ) #unique returns the first occurence (the largest value on the sorted list)
    
    best_matches = {x['point1']:x['point2'] for x in _edges[valid_index]}
    return best_matches

def _link_segments(_frame, _points, _points_g, _edges, _edges_g):
    edges_in_frame = _edges[_edges_g[_frame]]
    best_matches = _get_best_edge_match(edges_in_frame)
        
    segments_linked = []
    matched_points = set(best_matches.keys())
    points_in_frame = _points[[x for c in _points_g[_frame] for x in c]]
    points_in_frame_g = {x['point_id']:x for x in points_in_frame}
    
    for seg_id in range(n_segments):
        points_in_segment = _points[_points_g[_frame][seg_id]]
        
        prev_inds = {x[-1]['point_id']:x for x in segments_linked}
        matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
        matched_indices_cur = [best_matches[x] for x in matched_indices_prev]
        
        new_points = set(points_in_segment['point_id']) - set(matched_indices_cur)
        
        
        for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
            prev_inds[k1].append(points_in_frame_g[k2])
        segments_linked += [[points_in_frame_g[x]] for x in new_points]
    
    return segments_linked

def _join_segments_by_midpoint(segments2check):
    missing_segments = []
    skeletons = []
    
    matched_halves = defaultdict(list)
    for _half in segments2check:
        
        if len(_half) == n_segments:
            midbody_ind = _half[-1]['point_id']
            matched_halves[midbody_ind].append(_half)
        else:
            missing_segments.append(_half)
    
    
    for k, _halves in matched_halves.items():
        if len(_halves) == 2:
            skel = np.array(_halves[0] + _halves[1][-2::-1])
            skel = np.stack((skel['x'], skel['y']), axis=1)
            
            skeletons.append(skel)
        else:
            missing_segments += _halves
            
    return skeletons, missing_segments

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
    skel_size = 2*n_segments - 1
    
    tot_frames = points['frame_number'].max() + 1
    points_g = [[[] for _ in range(n_segments)] for _ in range(tot_frames)]
    for irow, row in enumerate(points):
        points_g[row['frame_number']][row['segment_id']].append(irow)
    
    edges_g = [[] for _ in range(tot_frames)]
    for irow, row in enumerate(edges):
        edges_g[row['frame_number']].append(irow)
        
    #%%
    
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    # for inds in points_g[frame]:
    #     for seg_id in range(n_segments):
    #         points_in_segment = points[points_g[frame][seg_id]]
    #         plt.plot(points_in_segment['x'], points_in_segment['y'], '.')
    #%%
    def _get_length(_skels):
        return np.linalg.norm(np.diff(_skels, axis=-2), axis=-1).sum(axis = -1)
    
    def _get_cost(src, target):
        rr = src[:, None] - target[None, ...]
        cost = np.nanmedian(np.sqrt((rr**2).sum(-1)), -1)
        return cost
    
    def _greedily_match(_cost, _threshold):
        #greedily match the points
        #assert _thresholds.size == _cost.shape[0]
        
        matches = []
        if np.all(_cost.shape):
            while True:
                ind_min = np.argmin(_cost)
                i_src , i_target = np.unravel_index(ind_min, _cost.shape)
                cur_cost = _cost[i_src , i_target]
                
                if (cur_cost >= _threshold) | (cur_cost != cur_cost):
                    break
                _cost[:, i_target] = _threshold
                _cost[i_src, :] = _threshold
                matches.append((i_src , i_target))
        return matches
        
    
    
    def _seglist2array(segments_list):
        segments_array = np.full((len(segments_list), n_segments, 2), np.nan)
        for iseg, seg in enumerate(segments_list):
            seg = np.array(seg)
            inds = seg['segment_id']
            segments_array[iseg, inds, 0] = seg['x']
            segments_array[iseg, inds, 1] = seg['y']
        
        return segments_array
        
        
    
    #max median dist
    #max_frac_dist = 2/n_segments
    max_frac_dist_half =  2/n_segments
    max_frac_dist_seg = 2/skel_size
    #midbody_dist = 
    
    skeletons = []
    missing_segments = []
    
    prev_frame_skeletons = []
    current_skel_id = 0
    #tot_frames
    for frame in tqdm.trange(tot_frames):
        segments_linked = _link_segments(frame, points, points_g, edges, edges_g)
        
        if not prev_frame_skeletons:
            matched_skeletons = []
            remaining_segments = segments_linked
        else:
            
            prev_skels = np.array([x[-1] for x in prev_frame_skeletons])
            prev_halves = np.concatenate((prev_skels[:, :n_segments], prev_skels[:, (n_segments-1):][:, ::-1]))
            
            segments2check = [x for x in segments_linked if len(x) > n_segments//2]
            short_segments = [x for x in segments_linked if len(x) < n_segments//2 ] 
            
            
            target_halves = _seglist2array(segments2check)
            
            match_threshold_half = max_frac_dist_half*np.median(_get_length(prev_halves))
            cost = _get_cost(prev_halves, target_halves)
            matches = _greedily_match(cost, match_threshold_half)
            
            n_prev_skels = len(prev_skels)
            prev_ids = [x[0] % n_prev_skels for x in matches]
            dat2match = {k : [-1, -1] for k in prev_ids}
            for pid, nid in matches:
                pid_r = pid % n_prev_skels
                half_id = pid//n_prev_skels
                dat2match[pid_r][half_id] = nid
            
            matched_skeletons = []
            remaining_inds = list(set(list(range(len(segments2check)))) - set([x[1] for x in matches]))
            for pid, (nid1, nid2) in dat2match.items():
                t1 = target_halves[nid1] if nid1 >= 0 else np.full((n_segments, 2), np.nan)
                t2 = target_halves[nid2] if nid2 >= 0 else np.full((n_segments, 2), np.nan)
                
                skel_n = np.concatenate((t1, t2[-2::-1]))
                matched_skeletons.append((pid, skel_n))
                
                # closest_dist = np.nanmin(np.linalg.norm(t1 - t2, axis=1))
                # if closest_dist <= 2*match_threshold:
                #     skel_n = np.concatenate((t1, t2[-2::-1]))
                # else:
                    
                #     plt.figure()
                    
                #     plt.figure()
                #     for skel in prev_skels:
                #         plt.plot(skel[:, 0], skel[:, 1], 'k')
                #     plt.plot(t1[:, 0], t1[:,1], '.-')
                #     plt.plot(t2[:, 0], t2[:,1], '.-')
                #     a
                    
                    
            remaining_segments = [segments2check[i] for i in remaining_inds]
            
            # i will only use complete segments to create a skeleton di novo
            remaining_segments = [x for x in remaining_segments if len(x) == n_segments] 
            short_segments += [x for x in remaining_segments if len(x) < n_segments]
            
            
            #try to match any remaining point
            unmatched_inds = set(list(range(n_prev_skels))) - set(dat2match.keys())
            unmatched_skeletons = [(i, _get_length(prev_skels[i]), prev_skels[i]) for i in unmatched_inds] 
            n_unmatched = len(unmatched_skeletons)
            
            skels2check = unmatched_skeletons
            #add matched with missing points
            for pid, skel_n in matched_skeletons:
                _bad = np.isnan(skel_n)
                if np.any(_bad):
                    skel_ = np.full_like(skel_n, np.nan)
                    skel_[_bad] = prev_skels[pid][_bad]
                    skels2check.append((pid, _get_length(prev_skels[pid]), skel_))
                  
            remaining_points = [p for x in short_segments for p in x]
            if skels2check and remaining_points:
                remaining_points = np.array(remaining_points)
                
                n2check = len(skels2check)
                prev_inds, skel_l, src_points = map(np.array, zip(*skels2check))
                src_points = src_points.reshape((-1, 2))
                
                dx = src_points[:, 0][:, None] - remaining_points['x'][None]
                dy = src_points[:, 1][:, None] - remaining_points['y'][None]
                cost = np.sqrt((dx*dx) + (dy*dy))
                
                
                #match_thresholds = max_frac_dist_seg*skel_l[:, None].repeat(skel_size, axis=1).reshape(-1)
                match_threshold_seg = max_frac_dist_seg*np.median(skel_l)
                matched_points = _greedily_match(cost, match_threshold_seg)
                
                if matched_points:
                    point_matched_skels = []
                    for ii, (pid, _, skel_) in enumerate(skels2check):
                        if ii < n_unmatched:
                            point_matched_skels.append((pid, np.full_like(skel_, np.nan)))
                        else:
                            point_matched_skels.append((pid, skel_))
                
                    src_ids, target_ids = zip(*matched_points)
                    ind1, ind2 = np.unravel_index(src_ids, (n2check, skel_size))
                    
                    for i1, i2, t in zip(ind1, ind2, target_ids):
                        p = remaining_points[t]
                        point_matched_skels[i1][-1][i2] = p['x'], p['y']
                    
                    matched_skeletons += point_matched_skels
                    
            
            matched_skeletons_r = matched_skeletons
            matched_skeletons = []
            for pid, skels_new in matched_skeletons_r:
                bad_ = np.isnan(skels_new[:, 0])
        
        
                if np.sum(bad_) > 2:
                    continue
                else:
                    skel_prev = prev_skels[pid]
                    skels_new[bad_] = skel_prev[bad_]
                    matched_skeletons.append((pid, skels_new))
                        
        
        
        
        new_skels, missing_in_frame = _join_segments_by_midpoint(remaining_segments)
        for skel in new_skels:
            skeletons.append([(frame, skel)])
        current_skel_id += len(new_skels)
        
        
        if len(matched_skeletons):
            
            matched_skels = np.array([x[-1] for x in matched_skeletons])
            largest_dists = np.max(np.linalg.norm(np.diff(matched_skels, axis=-2), axis=-1), axis=-1)
            for (pid, skel), largest_dist in zip(matched_skeletons, largest_dists):
                
                if largest_dist > match_threshold_half: 
                    #there is a segment with a very large separation. It is likely a bad skeleton
                    continue
                
                skel_id, _ = prev_frame_skeletons[pid]
                skeletons[skel_id].append((frame, skel))
            
            
        skels_in_frame = []
        for skel_id, traj_data in enumerate(skeletons):
            last_frame, last_skel = traj_data[-1]
            if last_frame == frame:
                skels_in_frame.append((skel_id, last_skel))
        
        #skeletons.append(skels_in_frame)
        #missing_segments.append(missing_in_frame)
        
        prev_frame_skeletons = skels_in_frame
                
        
    #%%
    frame = 15025
    with tables.File(masked_file, 'r') as fid:
        img = fid.get_node('/mask')[frame]
    
    plt.figure()
    plt.imshow(img, cmap ='gray')
    skels_in_frame = skels = [s[1] for ss in skeletons for s in ss if s[0] == frame]
    
    for skel in skels_in_frame:
        plt.plot(skel[:, 0], skel[:, 1], 'r-')
        
    
    
    #%%
    skeletons = sorted(skeletons, key = lambda x : x[0][0])
    initial_skels, final_skels = zip(*[((ii, *x[0]), (ii, *x[-1])) for ii, x in enumerate(skeletons)])
    initial_skels_d = defaultdict(list)
    for i_initial, (skel_id, frame, skel) in enumerate(initial_skels):
        initial_skels_d[frame].append((i_initial, skel_id, skel))
    initial_skels_d = {k : v for k, v in initial_skels_d.items()}
    
    max_gap = 12
    max_frac_dist = 0.25
    
    #join_costs_signed = np.full(( len(final_skels), len(initial_skels)), 1e10)
    skels2join = {}
    for i_final, (skel_id, frame, skel) in enumerate(final_skels):
        candidate_matches = []
        for f in range(frame + 1, frame + max_gap + 1):
            if f in initial_skels_d:
                candidate_matches += initial_skels_d[f]
        
        
        if not candidate_matches:
            continue
        
        i_initials, skel_targets_ids, skels_target = zip(*candidate_matches)
        skels_target = np.array(skels_target)
        
        cost1 = _get_cost(skel[None], skels_target)[0]/skel_size
        cost2 = _get_cost(skel[::-1][None], skels_target)[0]/skel_size
        w_cost_signed = np.where(cost2 < cost1, -cost2, cost1) #use it to flag a head/tail switch
        cost = np.abs(w_cost_signed)
        
        #greedy assigment 
        match_th = _get_length(skel)*max_frac_dist
        best_match = np.argmin(cost)
        
        if cost[best_match] < match_th:
            target_id = skel_targets_ids[best_match]
            target_id = (target_id, w_cost_signed[best_match]) #enconde head/tail switches as  negative indexes
            
            skels2join[skel_id] = target_id
        
    keys2check = list(skels2join.keys())
    
    joined_sequences = [[(keys2check.pop(0), None)]]
    
    while len(keys2check) > 0:
        seeded_list = joined_sequences[-1]
        seed, cost = seeded_list[-1]
        
        next_seed = skels2join[seed] #remember to abs since head/tail switches are encoded as negatives
        seeded_list.append(next_seed)
        
        r_next_seed = next_seed[0]
        if r_next_seed in keys2check:
            keys2check.remove(r_next_seed)
        else:
            joined_sequences.append([(keys2check.pop(0), None)])
    
    initial_seeds = [x[0][0] for x in joined_sequences]
    assert len(set(initial_seeds)) == len(initial_seeds)
    
    used_inds = set([x[0] for d in joined_sequences for x in d])
    remaining_inds = set(list(range(len(skeletons)))) - used_inds
    
    is_switch = False
    skels_joined = [skeletons[i] for i in remaining_inds]
    #skels_joined = []
    for inds2join in joined_sequences:
        ind, cost = inds2join[0]
        skels = skeletons[ind].copy()
        
        for ind, cost in inds2join[1:]:
            if cost < 0:
                is_switch = ~is_switch 
            
            next_skels = skeletons[ind]
            if is_switch:
                next_skels = [(t, s[::-1]) for t, s in next_skels]
            
            skels += next_skels
            
        is_switch = False
        skels_joined.append(skels)
    skels_joined = sorted(skels_joined, key = lambda x : x[0][0])
    #%%
    
    
    # # frame = 4934
    # # with tables.File(masked_file, 'r') as fid:
    # #     img = fid.get_node('/mask')[frame]
     
    # # plt.figure()
    # # plt.imshow(img, cmap ='gray')
    
    # # skels = [s[1] for ss in matched_skeletons for s in ss if s[0] == frame]
    # # for s in skels:
    # #     plt.plot(s[:, 0], s[:, 1], '-r')
    
    # # halves = missing_halves[frame]
    # # for s in halves:
    # #     plt.plot(s[:, 0], s[:, 1], '-c')
    
    # # ps = missing_points[frame]
    # # plt.plot(ps['x'], ps['y'], '.b')
    # #%%
    
    
    # max_frac_dist_seg = 2/skel_size
    # max_frac_dist_half = 0.5
    
    # skels = skels_joined[163]
    # last_frame, last_skel = skels[-1]
    # skel_length = _get_length(last_skel) 

    # next_frame = last_frame + 1
    
    # skels_new = np.full(last_skel.shape, np.nan)
            
    # halves2check = np.array(missing_halves[next_frame])
    # points2check = missing_points[next_frame]
    
    
    
        
    #     #p = points2check[i_match]
    #     #skels_new[inds2match[i_target], :] = (p['x'], p['y'])
    
    
    # if halves2check.shape[0] > 0:
    #     match_th_h = max_frac_dist_half*skel_length

    #     skel_halves = np.stack((last_skel[:n_segments], last_skel[n_segments-1:][::-1]))
    #     #skel_halves = _skel2halves(last_skel[None])
        
    #     cost = _get_cost(skel_halves, halves2check)
    #     target_inds, best_matches = _greedily_match(cost, match_th)
        
    #     #target_inds, best_matches = _get_best_matches(cost, match_th_h)
    #     #assert ((best_matches == 0) | (best_matches == 1)).all()
        
    #     for best_ind, t_ind in zip(best_matches, target_inds):
    #         assert (best_ind == 0) | (best_ind == 1)
    #         best_match_skel = halves2check[t_ind]
    #         if best_ind == 0:
    #             skels_new[:n_segments] = best_match_skel
    #         else:
    #             skels_new[n_segments-1:] = best_match_skel[::-1]
        
    # #221 400 # 395 226
    # if points2check.shape[0] > 0:
    #     match_th_p = max_frac_dist_seg*skel_length
        
    #     inds2match, = np.where(np.isnan(skels_new[:, 0]))
    #     points2match = last_skel[inds2match]
    #     dx = points2match[:, 0][:, None] - points2check['x'][None]
    #     dy = points2match[:, 1][:, None] - points2check['y'][None]
        
    #     cost = np.sqrt((dx*dx) + (dy*dy))
    #     best_matches, target_inds = _greedily_match(cost, match_th_p)
    #     if len(best_matches):
    #         matched_points = points2check[best_matches]
    #         matched_points = np.array((matched_points['x'], matched_points['y'])).T
    #         skels_new[inds2match[target_inds]] = matched_points
    
    #     # #greedily match the points
    #     # while True:
    #     #     ind_min = np.argmin(cost)
    #     #     i_target , i_match = np.unravel_index(ind_min, cost.shape)
    #     #     cur_cost = cost[i_target , i_match]
            
    #     #     if cur_cost >= match_th_p:
    #     #         break
    #     #     cost[:, i_match] = match_th_p
    #     #     cost[i_target, :] = match_th_p
            
    #     #     p = points2check[i_match]
    #     #     skels_new[inds2match[i_target], :] = (p['x'], p['y'])
            
            
    #     # best_matches, target_inds = _get_best_matches(cost, match_th_p)
    #     # if best_matches.size:
            
    #     #     matched_points = points2check[best_matches]
    #     #     matched_points = np.array((matched_points['x'], matched_points['y'])).T
    #     #     skels_new[inds2match[target_inds]] = matched_points
            
    # #bad_ = np.isnan(skels_new[:, 0])
    
    # import matplotlib.pylab as plt
    # plt.figure()
    # plt.plot(last_skel[:, 0], last_skel[:, 1], 'o-')
    # plt.plot(skels_new[:, 0], skels_new[:, 1], 's-')
    # plt.plot(points2check['x'], points2check['y'], 'x')
    # plt.plot(points2match[:, 0], points2match[:, 1], '.')
    # plt.axis('equal')
    # ll = 10
    # plt.xlim(np.min(last_skel[:, 0])- ll, np.max(last_skel[:, 0]) + ll)
    # plt.ylim(np.min(last_skel[:, 1])- ll, np.max(last_skel[:, 1]) + ll)
    # #

    #%%
    #skels_joined = skeletons
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
    tot_skeletons = sum([len(x) for x in skels_joined])
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
        
        for ii, worm_data in enumerate(tqdm.tqdm(skels_joined)):
            
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
            