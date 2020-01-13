#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:43:39 2019

@author: avelinojaver
"""
import tables
import pandas as pd
import numpy as np
from pathlib import Path
import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

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
        cost = np.nanmax(np.sqrt((rr**2).sum(-1)), -1)
        return cost
    
    def _greedily_match(_cost, _threshold):
        #greedily match the points
        #assert _thresholds.size == _cost.shape[0]
        
        matches = []
        if np.all(_cost.shape):
            while True:
                ind_min = np.nanargmin(_cost)
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
    max_frac_dist_seg = 0.5/skel_size
    
    skeletons = []
    missing_segments = []
    
    prev_frame_skeletons = []
    current_skel_id = 0
    #tot_frames
    
    
    weird_skels = []
    tot_frames
    for frame in tqdm.trange(tot_frames):
        segments_linked = _link_points_to_segments(frame, points, points_g, edges, edges_g, _link_points_to_segments)
        
        if not prev_frame_skeletons:
            matched_skeletons = []
            remaining_segments = segments_linked
        else:
            
            prev_skels = np.array([x[-1] for x in prev_frame_skeletons])
            prev_halves = np.concatenate((prev_skels[:, :n_segments], prev_skels[:, (n_segments-1):][:, ::-1]))
            
            segments2check = [x for x in segments_linked if len(x) > n_segments//2]
            short_segments = [x for x in segments_linked if len(x) <= n_segments//2 ] 
            
            target_halves = _seglist2array(segments2check)
            
            match_threshold_half = max(1, max_frac_dist_half*np.median(_get_length(prev_halves)))
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
            
            # plt.figure()
            # with tables.File(masked_file, 'r') as fid:
            #     img = fid.get_node('/mask')[frame]
            # plt.imshow(img, cmap ='gray')
            # for ss in segments_linked:
            #     ss = np.array(ss)
            #     plt.plot(ss['x'], ss['y'], 'r.-')
            
            # for skel in matched_skeletons.values():
            #     plt.plot(skel[:,0], skel[:, 1], 'b')
            # plt.xlim((0, 150))
            # plt.ylim((500, 650))
            
            used_inds = set([x[1] for x in matches])
            remaining_inds = set(list(range(len(segments2check)))) - used_inds
            remaining_segments = [segments2check[i] for i in remaining_inds]
            # i will only use complete segments to create a skeleton di novo
            short_segments += [x for x in remaining_segments if len(x) < n_segments]
            remaining_segments = [x for x in remaining_segments if len(x) == n_segments] 
            
            #try to match any remaining point
            unmatched_inds = set(list(range(n_prev_skels))) - set(dat2match.keys())
            unmatched_skeletons = [(i, _get_length(prev_skels[i]), prev_skels[i]) for i in unmatched_inds] 
            n_unmatched = len(unmatched_skeletons)
            
            points2check = unmatched_skeletons
            #add matched with missing points
            for pid, skel_n in matched_skeletons.items():
                _bad = np.isnan(skel_n)
                if np.any(_bad):
                    skel_ = np.full_like(skel_n, np.nan)
                    skel_[_bad] = prev_skels[pid][_bad]
                    points2check.append((pid, _get_length(prev_skels[pid]), skel_))
                
                  
            remaining_points = [p for x in short_segments for p in x]
            
            # plt.figure()
            # plt.imshow(img, cmap ='gray')
            # for skel in matched_skeletons.values():
            #     plt.plot(skel[:,0], skel[:, 1], 'b')
            
            # if remaining_points:
            #     p = np.array(remaining_points)
            #     plt.plot(p['x'], p['y'], 'r.')
            # for ss in remaining_segments:
            #     ss = np.array(ss)
            #     plt.plot(ss['x'], ss['y'], 'c.-')    
            # plt.xlim((0, 150))
            # plt.ylim((500, 650))
            
            # plt.figure()
            # skel = matched_skeletons[0]
            # plt.plot(skel[:,0], skel[:, 1], '.-b')
            
            if points2check and remaining_points:
                remaining_points_p = np.array(remaining_points)
                
                n2check = len(points2check)
                prev_inds, skel_l, src_points = map(np.array, zip(*points2check))
                src_points = src_points.reshape((-1, 2))
                
                dx = src_points[:, 0][:, None] - remaining_points_p['x'][None]
                dy = src_points[:, 1][:, None] - remaining_points_p['y'][None]
                cost = np.sqrt((dx*dx) + (dy*dy))
                
                #match_thresholds = max_frac_dist_seg*skel_l[:, None].repeat(skel_size, axis=1).reshape(-1)
                match_threshold_seg = max(1., max_frac_dist_seg*np.median(skel_l))
                matched_points = _greedily_match(cost, match_threshold_seg)
                
                if matched_points:
                    point_matched_skels = []
                    for ii, (pid, _, _) in enumerate(points2check):
                        if ii < n_unmatched:
                            point_matched_skels.append((pid, np.full_like(skel_, np.nan)))
                        else:
                            point_matched_skels.append((pid, matched_skeletons[pid]))
                
                    src_ids, target_ids = zip(*matched_points)
                    ind1, ind2 = np.unravel_index(src_ids, (n2check, skel_size))
                    
                    for i1, i2, t in zip(ind1, ind2, target_ids):
                        p = remaining_points[t]
                        point_matched_skels[i1][-1][i2] = p['x'], p['y']
                    
                    for pid,skel in point_matched_skels:
                        matched_skeletons[pid] = skel
            
            # skel = matched_skeletons[0]
            # plt.plot(skel[:,0], skel[:, 1], 'rv-')
            
            matched_skeletons_r = matched_skeletons
            matched_skeletons = {}
            for pid, skels_new in matched_skeletons_r.items():
                bad_ = np.isnan(skels_new[:, 0])
                if np.sum(bad_) > 4:
                    continue
                else:
                    skel_prev = prev_skels[pid]
                    skels_new[bad_] = skel_prev[bad_]
                    matched_skeletons[pid] =  skels_new
            # skel = matched_skeletons[0]
            # plt.plot(skel[:,0], skel[:, 1], 'r.-')
            
            
        new_skels, missing_in_frame = _join_segments_by_midpoint(remaining_segments)
        for skel in new_skels:
            skeletons.append([(frame, skel)])
        current_skel_id += len(new_skels)
        
        
        if len(matched_skeletons):
            matched_skels = np.array(list(matched_skeletons.values()))
            largest_dists = np.max(np.linalg.norm(np.diff(matched_skels, axis=-2), axis=-1), axis=-1)
            for (pid, skel), largest_dist in zip(matched_skeletons.items(), largest_dists):
                skel_id, _ = prev_frame_skeletons[pid]
                if largest_dist > 2*match_threshold_half: 
                    #there is a segment with a very large separation. It is likely a bad skeleton
                    weird_skels.append((largest_dist, skel_id, frame, skel))
                else:
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
    # for largest_dist, skel_id, frame, skel in weird_skels[-1:]:
    #     with tables.File(masked_file, 'r') as fid:
    #         img = fid.get_node('/mask')[frame]
    #     plt.figure()
    #     plt.imshow(img, cmap ='gray')
    #     plt.plot(skel[:, 0], skel[:, 1], 'r.-')
        
    #     ll = 10
    #     plt.xlim(np.min(skel[:, 0])- ll, np.max(skel[:, 0]) + ll)
    #     plt.ylim(np.min(skel[:, 1])- ll, np.max(skel[:, 1]) + ll)
    #%%
    # frame = 4983
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
    
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    # skels_in_frame = skels = [s[1] for ss in skeletons for s in ss if s[0] == frame]
    
    # for skel in skels_in_frame:
    #     plt.plot(skel[:, 0], skel[:, 1], 'r-')
    
    
    #%%
    # frame = 12440
    # segments_linked = _link_segments(frame, points, points_g, edges, edges_g)
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
    
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    # for ss in segments_linked:
    #     ss = np.array(ss)
    #     plt.plot(ss['x'], ss['y'], '.-')
    
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
        mean_angle = np.mean(angles)
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
    
    def correct_headtail(skels, fps = 25):
        
        skel_size = skels[0].shape[1]
        window_std = max(int(round(fps)), 5)
        segment4angle = max(1, int(round(skel_size / 10)))
        angles_head, angles_tail = _calculate_headtail_angles(skels, segment4angle)
        angles_head_ts = pd.Series(angles_head)
        angles_tail_ts = pd.Series(angles_tail)
        
        
        head_var = angles_head_ts.rolling(window_std).var()
        tail_var = angles_tail_ts.rolling(window_std).var()
        
        is_head_score = np.mean(head_var > tail_var)
        if is_head_score <= 0.5:
            skels = skels[:, ::-1]
        
        
        return skels, head_var, tail_var
    
    skeletons_oriented = []
    for skel_data in skeletons:
        frames, skels = map(np.array, zip(*skel_data))
        skels_corr = correct_headtail(skels)[0]
        skel_data_r = list(zip(frames, skels_corr))
        skeletons_oriented.append(skel_data_r)
    
    
    #%%
    frames, skels = map(np.array, zip(*skeletons_oriented[6]))
    skels, head_var, tail_var = correct_headtail(skels)
    
    fig, axs = plt.subplots(2, 1)
    #axs[0].plot(angles_head)
    #axs[0].plot(angles_tail)
    
    
    axs[1].plot(head_var)
    axs[1].plot(tail_var)
    
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
        initial_skels, final_skels = zip(*[((ii, *x[0]), (ii, *x[-1])) for ii, x in enumerate(skeletons)])
        
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
        #However,there is not a good official implementation (scipy)
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


    max_gap = 25#12
    max_frac_dist = 0.25
    
    skels2join = _match_trajectories(skeletons_oriented, 
                                     max_gap = max_gap, 
                                     max_frac_dist = max_frac_dist,
                                     include_switched_cost = False)
    joined_sequences = _get_joined_sequences(skels2join)
    
    initial_seeds = [x[0][0] for x in joined_sequences]
    assert len(set(initial_seeds)) == len(initial_seeds)
    
    used_inds = set([x[0] for d in joined_sequences for x in d])
    remaining_inds = set(list(range(len(skeletons_oriented)))) - used_inds
    
    is_switch = False
    skels_joined = [skeletons_oriented[i] for i in remaining_inds]
    for inds2join in joined_sequences:
        ind, cost = inds2join[0]
        skels = skeletons_oriented[ind].copy()
        
        for ind, cost in inds2join[1:]:
            if cost < 0:
                is_switch = ~is_switch 
            
            next_skels = skeletons_oriented[ind]
            if is_switch:
                next_skels = [(t, s[::-1]) for t, s in next_skels]
            
            skels += next_skels
            
        is_switch = False
        skels_joined.append(skels)

        
    # skels2join = _match_trajectories(skels_joined, 
    #                                  max_gap = max_gap, 
    #                                  max_frac_dist = max_frac_dist,
    #                                  include_switched_cost = True)
    # joined_sequences = _get_joined_sequences(skels2join)
    
    # initial_seeds = [x[0][0] for x in joined_sequences]
    # assert len(set(initial_seeds)) == len(initial_seeds)
    
    
    # skels_joined_o = skels_joined
    
    # used_inds = set([x[0] for d in joined_sequences for x in d])
    # remaining_inds = set(list(range(len(skels_joined_o)))) - used_inds
    
   
    # skels_joined = [skels_joined_o[i] for i in remaining_inds]
    # for inds2join in joined_sequences:
    #     ind, cost = inds2join[0]
    #     skels = skels_joined_o[ind].copy()
        
    #     for ind, cost in inds2join[1:]:
    #         if cost < 0:
    #             is_switch = ~is_switch 
            
    #         next_skels = skels_joined_o[ind]
    #         if is_switch:
    #             next_skels = [(t, s[::-1]) for t, s in next_skels]
            
    #         skels += next_skels
            
    #     is_switch = False
    #     skels_joined.append(skels)
        
    # skels_joined = sorted(skels_joined, key = lambda x : x[0][0])
    
    #%%
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    
    
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
            #this is to combat the error : "Expect x to be a 1-D sorted array_like." 
            #likely to appear because there are repeated values
            lengths, inds = np.unique(lengths, return_index=True)
            fx = interp1d(lengths, curve[inds, 0], kind = interp_kind)
            fy = interp1d(lengths, curve[inds, 1], kind = interp_kind)
            
    
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
    
    smooth_window_s = 0.25
    fps = 25
    target_n_segments = 49
    
    smooth_window = max(5, int(round(smooth_window_s*fps)))
    smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    
    
    smoothed_skeletons = []
    
    for worm_data in tqdm.tqdm(skels_joined):
        frames, skels = map(np.array, zip(*worm_data))
        
        #smooth skeletons
        for ii in range(skels.shape[0]):
            for icoord in range(2):
                skels[ii, :, icoord] = savgol_filter(skels[ii, :, icoord], 
                                                        window_length = 5, 
                                                        polyorder = 3,
                                                        mode = 'interp')
                
        #interpolate small gaps that occur during the join of trajectories
        expected_tot = frames[-1] - frames[0] + 1
        n_segments = skels.shape[1]
        if expected_tot != frames.size:
            frames_interp = np.arange(frames[0], frames[-1] + 1)
            skels_interp = np.full((expected_tot, n_segments, 2), np.nan)
            for iseg in range(n_segments):
                for icoord in range(2):
                    f = interp1d(frames, skels[:, iseg, icoord], kind = 'slinear')
                    skels_interp[:, iseg, icoord] = f(frames_interp)
            frames = frames_interp  
            skels = skels_interp
        
        #smooth on time
        if expected_tot > smooth_window:
            for iseg in range(n_segments):
                for icoord in range(2):
                    skels[:, iseg, icoord] = savgol_filter(skels[:, iseg, icoord], 
                                                               window_length = smooth_window, 
                                                               polyorder = 3,
                                                               mode = 'interp')
        
        #interpolate using a cubic spline to the 49 segments we typically use
        skels_smoothed = np.full((expected_tot, target_n_segments, 2), np.nan)
        for ii in range(expected_tot):
            skels_smoothed[ii] = _h_resample_curve(skels[ii], 
                                              resampling_N = target_n_segments, 
                                              interp_kind = 'cubic')[0]
        worm_data_s = list(zip(frames, skels_smoothed))
        smoothed_skeletons.append(worm_data_s)
        
    
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

    
    
    
    
    #%%
    #skels_joined = skeletons
    #skels2save = skeletons_oriented
    skels2save = skels_joined
    #skels2save = smoothed_skeletons
    
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
    
    skel_size = skels2save[0][0][-1].shape[0]
    
    curr_skel_id = 0
    tot_skeletons = sum([len(x) for x in skels2save])
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
                        shape = (tot_skeletons, skel_size, 2),
                        chunkshape = (1, skel_size, 2),
                        filters = TABLE_FILTERS
                        )
        
        for ii, worm_data in enumerate(tqdm.tqdm(skels2save)):
            
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
            