#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:43:39 2019

@author: avelinojaver
"""
import tables
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import math
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
    
def _get_best_edge_match(_edges, min_PAF = 0.2):
    _edges = _edges[_edges['cost_PAF'] >= min_PAF]
    _edges = _edges[np.argsort(_edges['cost_PAF'])[::-1]] #sort with the largest PAF first...
    _, valid_index =  np.unique(_edges['point1'], return_index = True ) #unique returns the first occurence (the largest value on the sorted list)
    
    best_matches = {x['point1']:x['point2'] for x in _edges[valid_index]}
    return best_matches

def _link_points_to_segments(_frame, _points, _points_g, _edges, _edges_g, n_segments):
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

def _join_segments_by_midpoint(segments2check, n_segments):
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

def _get_length(_skels):
    seg = np.linalg.norm(np.diff(_skels, axis=-2), axis=-1)
    return np.nansum(seg, axis = -1)
    
def _get_cost(src, target):
    rr = src[:, None] - target[None, ...]
    cost = np.nanmax(np.sqrt((rr**2).sum(-1)), -1)
    return cost

def _greedily_match(_cost, _threshold):
    #greedily match the points
    #assert _thresholds.size == _cost.shape[0]
    
    matches = []
    if np.all(_cost.shape):
        while True:
            if np.isnan(_cost).all():
                break
            
            ind_min = np.nanargmin(_cost)
            i_src , i_target = np.unravel_index(ind_min, _cost.shape)
            cur_cost = _cost[i_src , i_target]
            
            if (cur_cost >= _threshold) | (cur_cost != cur_cost):
                break
            
            _cost[:, i_target] = _threshold
            _cost[i_src, :] = _threshold
            matches.append((i_src , i_target))
    
    return matches
   
def _seglist2array(segments_list, n_segments):
    segments_array = np.full((len(segments_list), n_segments, 2), np.nan)
    for iseg, seg in enumerate(segments_list):
        seg = np.array(seg)
        inds = seg['segment_id']
        segments_array[iseg, inds, 0] = seg['x']
        segments_array[iseg, inds, 1] = seg['y']
    
    return segments_array

def _angles_btw_segments(skel):
    s1 = skel[:-2] - skel[1:-1]
    s2 = skel[2:] - skel[1:-1]
    
    #cos(theta) = v1.v2/(|v1||v2|)
    dotprod = np.sum(s1*s2, axis=1)
    s1_mag = np.linalg.norm(s1, axis=1)
    s2_mag = np.linalg.norm(s2, axis=1)
    theta = np.arccos(dotprod/(s1_mag*s2_mag))
    return theta
    
    

#%%
def link_skeletons(points, 
                   edges, 
                   n_segments = 8, 
                   max_frac_dist_half = None,
                   max_frac_dist_seg = None
                   ):
    check_frame_window = 10
    skel_size = 2*n_segments - 1
    
    if max_frac_dist_half is None:
        max_frac_dist_half =  2/n_segments #2/n_segments
    if max_frac_dist_seg is None:
        max_frac_dist_seg = 1/n_segments #0.5/skel_size #1/skel_size
    
    tot_frames = points['frame_number'].max() + 1
    
    tot_points = points['segment_id'].max() + 1
    has_head_score = tot_points == (n_segments + 1) # I am expecting an extra map to show head tail orientation
        
    
    points_g = [[[] for _ in range(tot_points)] for _ in range(tot_frames)]
    
    
    for irow, row in enumerate(points):
        points_g[row['frame_number']][row['segment_id']].append(irow)
    
    edges_g = [[] for _ in range(tot_frames)]
    for irow, row in enumerate(edges):
        edges_g[row['frame_number']].append(irow)
       
    skeletons = []
    prev_frame_skeletons = []
    weird_skels = []
    
    current_skel_id = 0
    #tot_frames = 1000
    for frame in tqdm.trange(tot_frames):
        segments_linked = _link_points_to_segments(frame, points, points_g, edges, edges_g, n_segments)
        
        if not prev_frame_skeletons:
            matched_skeletons = []
            
            
            remaining_segments = segments_linked
        else:
            
            #### match previous skeletons to their halves using large sements from the current frame
            prev_skels = np.array([x[-1] for x in prev_frame_skeletons])
            prev_halves = np.concatenate((prev_skels[:, :n_segments], prev_skels[:, (n_segments-1):][:, ::-1]))
            
            segments2check = [x for x in segments_linked if len(x) > n_segments//2]
            short_segments = [x for x in segments_linked if len(x) <= n_segments//2 ] 
            
            target_halves = _seglist2array(segments2check, n_segments)
            
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
            
            matched_skeletons = {}
            for pid, (nid1, nid2) in dat2match.items():
                t1 = target_halves[nid1] if nid1 >= 0 else np.full((n_segments, 2), np.nan)
                t2 = target_halves[nid2] if nid2 >= 0 else np.full((n_segments, 2), np.nan)
                
                skel_n = np.concatenate((t1, t2[-2::-1]))
                matched_skeletons[pid] = skel_n
            
            ##### try to match any remaining predicted points that have not been used yet
            # first localize unmatched segments
            used_inds = set([x[1] for x in matches])
            remaining_inds = set(list(range(len(segments2check)))) - used_inds
            remaining_segments = [segments2check[i] for i in remaining_inds]
            
            # i want to only use complete segments to create a skeleton di novo
            short_segments += [x for x in remaining_segments if len(x) < n_segments]
            remaining_segments = [x for x in remaining_segments if len(x) == n_segments] 
            
            #try to match any remaining point
            unmatched_inds = set(list(range(n_prev_skels))) - set(dat2match.keys())
            unmatched_skeletons = [(i, _get_length(prev_skels[i]), prev_skels[i]) for i in unmatched_inds] 
            n_unmatched = len(unmatched_skeletons)
            
            points2check = unmatched_skeletons
            #add to the list to check any matched skeletons with missing points, i.e only one half of the skeleton was found, or the assigned segment have missing points
            for pid, skel_n in matched_skeletons.items():
                _bad = np.isnan(skel_n)
                if np.any(_bad):
                    skel_ = np.full_like(skel_n, np.nan)
                    skel_[_bad] = prev_skels[pid][_bad]
                    points2check.append((pid, _get_length(prev_skels[pid]), skel_))
            remaining_points = [p for x in short_segments for p in x]
            
            
            
            if points2check and remaining_points:
                #match remaing points with the missing points in the skeletons
                remaining_points_p = np.array(remaining_points)
                
                n2check = len(points2check)
                prev_inds, skel_l, src_points = map(np.array, zip(*points2check))
                src_points = src_points.reshape((-1, 2))
                
                dx = src_points[:, 0][:, None] - remaining_points_p['x'][None]
                dy = src_points[:, 1][:, None] - remaining_points_p['y'][None]
                cost = np.sqrt((dx*dx) + (dy*dy))
                
                #match_thresholds = max_frac_dist_seg*skel_l[:, None].repeat(skel_size, axis=1).reshape(-1)
                match_threshold_seg = max_frac_dist_seg*np.median(skel_l)
                matched_points = _greedily_match(cost, match_threshold_seg)
                
                if matched_points:
                    point_matched_skels = []
                    for ii, (pid, _, _) in enumerate(points2check):
                        if ii < n_unmatched:
                            point_matched_skels.append((pid, np.full((skel_size, 2), np.nan)))
                        else:
                            point_matched_skels.append((pid, matched_skeletons[pid]))
                
                    src_ids, target_ids = zip(*matched_points)
                    ind1, ind2 = np.unravel_index(src_ids, (n2check, skel_size))
                    
                    for i1, i2, t in zip(ind1, ind2, target_ids):
                        p = remaining_points[t]
                        point_matched_skels[i1][-1][i2] = p['x'], p['y']
                    
                    for pid,skel in point_matched_skels:
                        matched_skeletons[pid] = skel
            
            #if a matched skeleton has a lot of missing points (n_segments/2) it will be removed
            #In skeletons with only a few missing points, the missing points 
            #will be replaced by the corresponding points in from the matched skeleton in the previous frame
            matched_skeletons_r = matched_skeletons
            matched_skeletons = {}
            for pid, skels_new in matched_skeletons_r.items():
                bad_ = np.isnan(skels_new[:, 0])
                if np.sum(bad_) > n_segments//2:
                    continue
                else:
                    #skel_prev = prev_skels[pid]
                    #skels_new[bad_] = skel_prev[bad_]
                    matched_skeletons[pid] =  skels_new
            
        #I will attempt to join remaining segments by matching their middle point. 
        #This skeletons correspond to the new skeletons (new trajectories).
        new_skels, missing_in_frame = _join_segments_by_midpoint(remaining_segments, n_segments)
        for skel in new_skels:
            skeletons.append([(frame, skel)])
        current_skel_id += len(new_skels)
        
        #if frame == 6267:
        #    import pdb
        #    pdb.set_trace()
        
        #%%
        #append the matched skeletons to their corresponding list. 
        if len(matched_skeletons):
            for pid, skel in matched_skeletons.items():
                skel_id, _ = prev_frame_skeletons[pid]
                theta = _angles_btw_segments(skel)
                
                
                segment_sizes = np.linalg.norm(np.diff(skel, axis=-2), axis = -1)
                largest_size = np.nanmax(segment_sizes)
                median_size = np.nanmedian(segment_sizes)
                
                
                #Any segment cannot be more than three times the median distance between segments 
                #or twice the threshold to match halfs of worms
                size_th = min(2*match_threshold_half, 3*median_size)
                
                #This is an extra step to ensure quality of the skeleton.
                #If there is a segment with a very large separation. It is likely a bad skeleton
                if (largest_size > size_th)  | np.any(theta < np.pi/2): 
                    weird_skels.append((skel_id, frame, skel))
                else:
                    skeletons[skel_id].append((frame, skel))
        
        
        skels_in_frame = []
        for skel_id, traj_data in enumerate(skeletons):
            last_frame, last_skel = traj_data[-1]
            
            if frame - last_frame < check_frame_window:
                skels_in_frame.append((skel_id, last_skel))
        
        prev_frame_skeletons = skels_in_frame    
    
    #return skeletons
    
    skels_with_head_scores = []
    head_score = 0
    for traj_data in skeletons:
        traj_data_h = []
        for frame, skel in traj_data:
            if has_head_score:
                ht_points = points[points_g[frame][-1]]
                head_score = _get_head_tail_score(skel, ht_points)
            traj_data_h.append((frame, skel, head_score))
        skels_with_head_scores.append(traj_data_h)
    
    return skels_with_head_scores

def _get_head_tail_score(skel, ht_points, head_ind = 2, max_dist = 2):
    
    tail_id = skel.shape[0]-(head_ind + 1)
    
    dx = (skel[head_ind, 0] - ht_points['x'])
    dy = (skel[head_ind, 1] - ht_points['y'])
    r_head = np.sqrt(dx**2 + dy**2).min()
    
    dx = (skel[tail_id, 0] - ht_points['x'])
    dy = (skel[tail_id, 1] - ht_points['y'])
    r_tail = np.sqrt(dx**2 + dy**2).min()
    
    if r_head > max_dist:
        return 0 if r_tail > max_dist else -1
    else:
        return 1 if r_tail > max_dist else 0
    
        
    
#%%
def _match_trajectories(skeletons, max_gap, max_frac_dist, include_switched_cost = True):
    '''
    returns: 
        skels2join => dictionary k = previous trajectory index, v = (next trajectory index, signed cost of linkage )
        a negative linkage cost correspond to a head/tail switch
    '''
    
    #first sort the skeletons by time
    skeletons = sorted(skeletons, key = lambda x : x[0][0])
    
    #list of the first and last skeletons of each trajectory
    initial_skels, final_skels = zip(*[((ii, *x[0][:2]), (ii, *x[-1][:2])) for ii, x in enumerate(skeletons)])
    
    #group the initial trajectories by frame. I will use this to locate matching candidates
    initial_skels_d = defaultdict(list)
    for i_initial, (skel_id, frame, skel) in enumerate(initial_skels):
        initial_skels_d[frame].append((i_initial, skel_id, skel))
    initial_skels_d = {k : v for k, v in initial_skels_d.items()}
    
    #build the cost matrix
    largest_val = 1e6
    join_costs_signed = np.full(( len(final_skels), len(initial_skels)), largest_val)
    for i_final, (skel_id, frame, skel) in enumerate(final_skels):
        #candidate matches are trajectories that initiate within a `max_gap` frames
        # of the end of the trajectory to match
        candidate_matches = []
        for f in range(frame + 1, frame + max_gap + 1):
            if f in initial_skels_d:
                candidate_matches += initial_skels_d[f]
        
        if not candidate_matches:
            continue
        
        i_initials, skel_targets_ids, skels_target = zip(*candidate_matches)
        skels_target = np.array(skels_target)
        
        cost1 = _get_cost(skel[None], skels_target)[0]
        if include_switched_cost:
            #i calculate one cost of the match using the normal and a switched version 
            #of the skeleton to accound for head/ tail switches.
            
            cost2 = _get_cost(skel[::-1][None], skels_target)[0]
            
            #I will assign a negative value as a flag if there is a head/tail switch.
            #however the linear assigment will occur in the absolute values
            w_cost_signed = np.where(cost2 < cost1, -cost2, cost1) 
        else:
            w_cost_signed = cost1
        
        #any cost above the match threshold will be assigned to the largest value
        #this will effectively remove the values from the minimization procedure
        match_th = _get_length(skel)*max_frac_dist
        w_cost_signed[np.abs(w_cost_signed)>match_th] = largest_val 
        
        join_costs_signed[i_final, i_initials] = w_cost_signed
    
    #for speed reasons i want to make the assigment to only the columns with a valid match candidate
    bad = join_costs_signed >= largest_val
    valid_final = ~bad.all(axis=1)
    valid_initial = ~bad.all(axis=0)
    
    initial_inds = np.array([x[0] for x in initial_skels])
    initial_inds = initial_inds[valid_initial]
    final_inds = np.array([x[0] for x in final_skels])
    final_inds = final_inds[valid_final]
    
    concat_costs_signed = join_costs_signed[valid_final, :][:, valid_initial]
    compact_cost = np.abs(concat_costs_signed)
    
    #finally we do the linear assigment. If the matrix is very lage this could take forever.
    #for the moment it seems it is working fine, but it might be good to replace 
    #it with something like lapjv (https://github.com/berhane/LAP-solvers). 
    #However,there is not official implementation (scipy)
    final_ind, ini_ind = linear_sum_assignment(compact_cost)
    _good = ~(compact_cost[final_ind, ini_ind] >= largest_val)
    final_ind, ini_ind = final_ind[_good], ini_ind[_good]
    
    skels2join = {final_inds[k] : (initial_inds[v], concat_costs_signed[k,v]) for k, v in zip(final_ind, ini_ind)}
    assert all([k != v[0] for k, v in  skels2join.items()])
    
    return skels2join

def _get_joined_sequences(skels2join):
    '''
    Get a list of the trajectories to be joined
    '''
    keys2check = list(skels2join.keys())
    if not keys2check:
        return []
    
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
    return joined_sequences

def join_trajectories(skeletons,
                      max_gap = 12,
                      max_frac_dist = 0.25):
    
    skels2join = _match_trajectories(skeletons, max_gap = max_gap, max_frac_dist = max_frac_dist)
    joined_sequences = _get_joined_sequences(skels2join)
    
    initial_seeds = [x[0][0] for x in joined_sequences]
    assert len(set(initial_seeds)) == len(initial_seeds)
    
    used_inds = set([x[0] for d in joined_sequences for x in d])
    remaining_inds = set(list(range(len(skeletons)))) - used_inds
    
    is_switch = False
    skels_joined = [skeletons[i] for i in remaining_inds]
    for inds2join in joined_sequences:
        ind, cost = inds2join[0]
        skels = skeletons[ind].copy()
        
        for ind, cost in inds2join[1:]:
            if cost < 0:
                is_switch = ~is_switch 
            
            next_skels = skeletons[ind]
            if is_switch:
                next_skels = [(t, s[::-1], h) for t, s, h in next_skels]
            
            skels += next_skels
            
        is_switch = False
        skels_joined.append(skels)
    skels_joined = sorted(skels_joined, key = lambda x : x[0][0])
    return skels_joined
#%%
def _h_resample_curve(curve, resampling_N=49, widths=None, interp_kind = 'linear'):
    '''Resample curve to have resampling_N equidistant segments
    I give width as an optional parameter since I want to use the 
    same interpolation as with the skeletons
    
    I calculate the length here indirectly
    '''
    
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)
    
    _valid = dr > 0 #remove repeated values
    
    if not np.all(_valid):
        dr = dr[_valid]
        curve = curve[np.hstack((True, _valid))]
    
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]

    # Verify array lengths
    if len(lengths) < 2 or len(curve) < 2:
        return None, None, None
    try:
        fx = interp1d(lengths, curve[:, 0], kind = interp_kind)
        fy = interp1d(lengths, curve[:, 1], kind = interp_kind)
    except ValueError:
        import pdb
        pdb.set_trace()
        

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    # I add the epsilon because otherwise the interpolation will produce nan
    # for zero
    try:
        resampled_curve = np.zeros((resampling_N, 2))
        resampled_curve[:, 0] = fx(subLengths)
        resampled_curve[:, 1] = fy(subLengths)
        if widths is not None:
            fw = interp1d(lengths, widths)
            widths = fw(subLengths)
    except ValueError:
        resampled_curve = np.full((resampling_N, 2), np.nan)
        widths = np.full(resampling_N, np.nan)

    return resampled_curve, tot_length, widths


def _get_group_borders(is_valid, pad_val = False):
    
    #add zeros at the edge to consider any block in the edges
    index = np.hstack([pad_val, is_valid , pad_val])
    switches = np.diff(index.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)
    assert turn_off.size == turn_on.size
    
    #fin if fin<index.size else fin-1)
    ind_ranges = list(zip(turn_on, turn_off))
    return ind_ranges
#%%
def _fill_small_gaps(is_valid, max_gap_size):
    ind_ranges = _get_group_borders(~is_valid)
    #ifter by the gap size
    ind_ranges = [(ini, fin) for ini, fin in ind_ranges if fin-ini > max_gap_size]
    
    index_filled = np.ones_like(is_valid)
    for ini, fin in ind_ranges:
        index_filled[ini:fin+1] = False

    return index_filled  
#%%

def interpolate_and_smooth(skels_o, frames_o, smooth_window, interp_max_gap):
    
    frames_inds = frames_o - frames_o[0]
    expected_tot = frames_inds[-1] + 1
    n_segments = skels_o.shape[1]
    
    skels2fill = np.full((expected_tot, n_segments, 2), np.nan)
    skels2fill[frames_inds] = skels_o
    
    is_valid = ~(np.isnan(skels2fill).any(axis=-1))
    is_valid_filled = is_valid.copy()
    for ii in range(is_valid.shape[1]):
        is_valid_filled[:, ii] = _fill_small_gaps(is_valid[:, ii], interp_max_gap)
    is_valid_filled = is_valid_filled.all(axis=-1)
    
    
    if np.sum(is_valid.all(axis=1)) <= 3:
        return np.zeros((0, n_segments, 2), dtype = skels_o.dtype), np.zeros(0, dtype = frames_o.dtype)
    
    frames_interp = np.arange(frames_inds[0], frames_inds[-1] + 1)
    if is_valid.all():
        assert skels2fill.shape[0] == skels_o.shape[0]
        skels = skels2fill
        frames = frames_o.copy()
    else:
        skels_interp = np.full((expected_tot, n_segments, 2), np.nan)
        for iseg in range(n_segments):
            for icoord in range(2):
                yy = skels2fill[:, iseg, icoord]
                
                _valid = ~np.isnan(yy)
                
                if not np.all(_valid):
                    f = interp1d(frames_interp[_valid], 
                                 yy[_valid], kind = 
                                 'linear', 
                                 bounds_error = False,
                                 fill_value = 'extrapolate'
                                 )
                    skels_interp[:, iseg, icoord] = f(frames_interp)
                else:
                    skels_interp[:, iseg, icoord] = yy
        
        skels = skels_interp
        frames = frames_interp + frames_o[0]
    
    #smooth on time
    if skels.shape[0] > smooth_window:
        for iseg in range(n_segments):
            for icoord in range(2):
                skels[:, iseg, icoord] = savgol_filter(skels[:, iseg, icoord], 
                                                            window_length = smooth_window, 
                                                            polyorder = 3,
                                                            mode = 'interp'
                                                            )
    
    skels = skels[is_valid_filled]
    frames = frames[is_valid_filled]
    
    # dd = np.diff(frames)
    # if ((dd > 1) & (dd <interp_max_gap)).any():
    #     import pdb
    #     pdb.set_trace()
    
    return skels, frames
    


def smooth_skeletons(skeletons, 
                     smooth_window_s = 0.25,
                     interp_max_gap_s = 0.25,
                     fps = 25,
                     target_n_segments = 49
                     ):
    
    smooth_window = max(5, int(round(smooth_window_s*fps)))
    smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    interp_max_gap = max(1, int(round(interp_max_gap_s*fps)))
    
    smoothed_skeletons = []
    
    for worm_data in tqdm.tqdm(skeletons):
        frames, skels = map(np.array, zip(*worm_data))
        skels, frames = interpolate_and_smooth(skels, frames, smooth_window, interp_max_gap)
        
        
        #skels_smoothed = skels
        #interpolate using a cubic spline to the 49 segments we typically use
        skels_smoothed = np.full((skels.shape[0], target_n_segments, 2), np.nan)
        for ii, skel in enumerate(skels):
            skels_smoothed[ii] = _h_resample_curve(skel, 
                                              resampling_N = target_n_segments, 
                                              interp_kind = 'cubic')[0]
        
        
        
        worm_data_s = list(zip(frames, skels_smoothed))
        
        if not len(worm_data_s):
            continue
        
        smoothed_skeletons.append(worm_data_s)
        
        
    return smoothed_skeletons

#%%
def _get_angles_delta(dx, dy):
    '''
    Calculate angles and fix for any jump between -2pi to 2pi
    '''
    angles = np.arctan2(dx, dy)
    d_ang = np.diff(angles)

    # %+1 to cancel shift of diff
    positive_jumps = np.where(d_ang > np.pi)[0] + 1
    negative_jumps = np.where(d_ang < -np.pi)[0] + 1

    #% subtract 2pi from remainging data after positive jumps
    for jump in positive_jumps:
        angles[jump:] = angles[jump:] - 2 * np.pi

    #% add 2pi to remaining data after negative jumps
    for jump in negative_jumps:
        angles[jump:] = angles[jump:] + 2 * np.pi

    #% rotate skeleton angles so that mean orientation is zero
    mean_angle = np.nanmean(angles)
    angles = angles - mean_angle

    return angles, mean_angle

def _calculate_headtail_angles(skeletons, segment4angle, good = None):
    '''
    For each skeleton two angles are caculated: one vector between the index 0 and segment4angle ('head'), and the other from the index -1 and -segment4angle-1 ('tail').
    '''
    tot = skeletons.shape[0]
    
    
    if good is not None:
        skeletons = skeletons[good]
    
    dx_h = skeletons[:, segment4angle, 0] - skeletons[:, 0, 0]
    dy_h = skeletons[:, segment4angle, 1] - skeletons[:, 0, 1]

    dx_t = skeletons[:, - segment4angle - 1, 0] - skeletons[:, -1, 0]
    dy_t = skeletons[:, - segment4angle - 1, 1] - skeletons[:, -1, 1]
    
    ang_h, _ = _get_angles_delta(dx_h, dy_h)
    ang_t, _ = _get_angles_delta(dx_t, dy_t)
    
    if good is None:
        angles_head, angles_tail = ang_h, ang_t
    else:
        angles_head = np.full(tot, np.nan)
        angles_tail = np.full(tot, np.nan)
        
        angles_head[good] = ang_h
        angles_tail[good] = ang_t
    
    return angles_head, angles_tail

def _correct_ht_by_movement(skels, fps):
    
    skel_size = skels[0].shape[1]
    window_std = max(int(round(fps)), 5)
    segment4angle = max(1, int(math.ceil(skel_size / 10)))
    angles_head, angles_tail = _calculate_headtail_angles(skels, segment4angle)
    angles_head_ts = pd.Series(angles_head)
    angles_tail_ts = pd.Series(angles_tail)
    
    
    head_var = angles_head_ts.rolling(window_std).apply(np.nanstd)
    tail_var = angles_tail_ts.rolling(window_std).apply(np.nanstd)
    
    is_head_score = np.mean(head_var > tail_var)
    if is_head_score <= 0.5:
        skels = skels[:, ::-1]
    
    return skels
def correct_headtail(skeletons, fps):
    
    skeletons_oriented = []
    for skel_data in skeletons:
        frames, skels, head_scores = map(np.array, zip(*skel_data))
        #import pdb
        #pdb.set_trace()
        
        skels_corr = _correct_ht_by_movement(skels, fps)
        skel_data_r = list(zip(frames, skels_corr))
        skeletons_oriented.append(skel_data_r)
        
    return skeletons_oriented



#%%
    
def save_skeletons(skeletons_data, save_file):
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
    
    
    if not skeletons_data:
        return
    
    
    skel_size = skeletons_data[0][0][1].shape[0]
    
    curr_skel_id = 0
    tot_skeletons = sum([len(x) for x in skeletons_data])
    with tables.File(str(save_file), 'w') as fid:
        
        
        inds_tab = fid.create_table('/',
                    "trajectories_data",
                    tables_dtypes,
                    filters = TABLE_FILTERS
                    )
        
        w_node = fid.create_group('/', 'coordinates')
        skel_array = fid.create_carray(w_node, 
                        'skeletons',
                        atom = tables.Float32Atom(),
                        shape = (tot_skeletons, skel_size, 2),
                        chunkshape = (1, skel_size, 2),
                        filters = TABLE_FILTERS
                        )
        
        for ii, worm_data in enumerate(tqdm.tqdm(skeletons_data)):
            _, skels = zip(*worm_data)
            skels = np.array(skels)
            
            
            coord_min = np.nanmin(skels, axis=1)
            coord_max = np.nanmax(skels, axis=1)
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
    


def _process_file_single_worm(unlinked_segments_file, 
                  save_file, 
                  n_segments = 8, 
                  max_gap_btw_traj = 12,
                  fps = 25,
                  smooth_window_s = 0.25,
                  target_n_segments = 49
                  ):
    
    with tables.File(unlinked_segments_file) as fid:
        points = fid.get_node('/points')[:]
        edges = fid.get_node('/edges')[:]
    
    skeletons = link_skeletons(points, edges, n_segments = n_segments)
    skeletons = join_trajectories(skeletons, max_gap = max_gap_btw_traj, max_frac_dist = 2/n_segments)
    skeletons = correct_headtail(skeletons, fps)
    
    skeletons = smooth_skeletons(skeletons, 
                                      smooth_window_s = smooth_window_s, 
                                      fps = fps, 
                                      target_n_segments = target_n_segments)
    
    save_skeletons(skeletons, save_file) 
    return skeletons
if __name__ == '__main__':
    import random
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/mating')
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/Results')
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/WT2')
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Serena_WT_Screening')
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Syngenta')
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Pratheeban')
    
    assert root_dir.exists()
    #unlinked_segments_file = root_dir / (bn + '_unlinked-skels-rois.hdf5')
    #skel_file = unlinked_file = root_dir / (bn + '_skeletonsNN.hdf5')
    #_process_file(unlinked_segments_file, skel_file)
    
    _ext = '_unlinked-skels.hdf5'
    ext2save = '_skeletonsNN.hdf5'
    #files2process = list(root_dir.rglob('*' + _ext))
    files2process = list(root_dir.glob('*' + _ext))
    
    #files2process = list(root_dir.rglob('JU792_Ch1_24092017_063115' + _ext))
    
    random.shuffle(files2process)
    for fname in tqdm.tqdm(files2process):
        save_file = fname.parent / fname.name.replace(_ext, ext2save)
        if save_file.exists():
            continue
        
        skeletons = _process_file_single_worm(fname, save_file)
        
    #%%
    
    
    
       