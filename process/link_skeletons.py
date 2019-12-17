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
    skel_size = 2*n_segments - 1
    
    tot_frames = points['frame_number'].max() + 1
    points_g = [[[] for _ in range(n_segments)] for _ in range(tot_frames)]
    for irow, row in enumerate(points):
        points_g[row['frame_number']][row['segment_id']].append(irow)
    
    edges_g = [[] for _ in range(tot_frames)]
    for irow, row in enumerate(edges):
        edges_g[row['frame_number']].append(irow)
        
    #%%
    # frame = 48
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
     
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    # for inds in points_g[frame]:
    #     for seg_id in range(n_segments):
    #         points_in_segment = points[points_g[frame][seg_id]]
    #         plt.plot(points_in_segment['x'], points_in_segment['y'], '.')
    
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
        
        if skeletons_in_frames:
            skeletons_in_frames = np.array(skeletons_in_frames)
        else:
            skeletons_in_frames = np.zeros((0, skel_size, 2))
        skeletons.append(skeletons_in_frames)
        
        if missing_halves_in_frame:
            missing_halves_in_frame = np.array(missing_halves_in_frame)
        else:
            missing_halves_in_frame = np.zeros((0, n_segments, 2))
        missing_halves.append(missing_halves_in_frame)
        
        missing_in_frame = np.array([points_in_frame_g[p] for p in bad_index])
        missing_points.append(missing_in_frame)
        
    #%%
    # frame = 48
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
     
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    
    # skels = skeletons[frame]
    # for s in skels:
    #     plt.plot(s[:, 0], s[:, 1], '-r')
    
    # halves = missing_halves[frame]
    # for s in halves:
    #     plt.plot(s[:, 0], s[:, 1], '-c')
    
    # ps = missing_points[frame]
    # plt.plot(ps['x'], ps['y'], '.b')
    
    
    #%% #link on time
    def _skel2halves(_skels, n_segments = 8):
        return np.concatenate((_skels[:, :n_segments], _skels[:, (n_segments-1):][:, ::-1]))
    
    def _get_cost(halves1, halves2):
        rr = halves1[:, None] - halves2[None, ...]
        cost = np.median(np.sqrt((rr**2).sum(-1)), -1)
        return cost
    
    def _get_length(_skels):
        return np.sqrt(np.diff(_skels, axis=-2)**2).sum(-1).sum(-1)
    
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
            prev_length = _get_length(prev_skels) 
            
            prev_half = _skel2halves(prev_skels)
            curr_half = _skel2halves(skels)
            
            cost = _get_cost(prev_half, curr_half)
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
    #%%
    # frame = 4945
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
     
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    
    # skels = [s[1] for ss in matched_skeletons for s in ss if s[0] == frame]
    # for s in skels:
    #     plt.plot(s[:, 0], s[:, 1], '-r')
    
    # halves = missing_halves[frame]
    # for s in halves:
    #     plt.plot(s[:, 0], s[:, 1], '-c')
    
    # ps = missing_points[frame]
    # plt.plot(ps['x'], ps['y'], '.b')
    #%%
    # def _get_cost(halves1, halves2):
    #     rr = halves1[:, None] - halves2[None, ...]
    #     cost = np.sqrt((rr**2).sum(-1)).sum(-1)
    #     return cost
    
    max_frac_dist_half = 0.5
    max_frac_dist_seg = 1/skel_size
    
    #matched_skeletons = matched_skeletons_bkp
    matched_skeletons = sorted(matched_skeletons, key = lambda x : len(x), reverse = True)
    #matched_skeletons_bkp = [x.copy() for x in matched_skeletons]
    #%%
    # def _get_best_matches(cost, match_th):
    #     inds, best_match = linear_sum_assignment(cost)
        
    #     #this is important I want all the weights of invalid points to have the same cost. 
    #     #Otherwise the algorith will try to minimize them and could give weird answers.
    #     cost[cost>match_th] = match_th 
    #     cost_matches = cost[inds, best_match]
        
    #     _valid = cost_matches<match_th
    #     best_matches = best_match[_valid]
    #     inds = inds[_valid]
        
    #     return best_matches, inds
    
    
    def _greedily_match(cost, match_th):
        #greedily match the points
        matches = []
        if np.all(cost.shape):
            while True:
                ind_min = np.argmin(cost)
                i_target , i_match = np.unravel_index(ind_min, cost.shape)
                cur_cost = cost[i_target , i_match]
                
                if cur_cost >= match_th:
                    break
                cost[:, i_match] = match_th
                cost[i_target, :] = match_th
                matches.append((i_match, i_target))
        
        if not matches:
            return np.zeros(0), np.zeros(0)
        else:
            best_matches, target_matches = map(np.array, zip(*matches))
            return best_matches, target_matches
    
    missing_halves_r = missing_halves.copy()
    missing_points_r = missing_points.copy()
    
    for skels in tqdm.tqdm(matched_skeletons):#[2187:2188]):
        while True:
            last_frame, last_skel = skels[-1]
            skel_length = _get_length(last_skel) 
        
            next_frame = last_frame + 1
            if next_frame >= tot_frames:
                break
            
            skels_new = np.full(last_skel.shape, np.nan)
            
            halves2check = np.array(missing_halves_r[next_frame])
            points2check = missing_points_r[next_frame]
            
            if halves2check.shape[0] > 0:
                match_th_h = max_frac_dist_half*skel_length
        
                
                skel_halves = _skel2halves(last_skel[None])
                
                
                cost = _get_cost(halves2check, skel_halves)
                best_matches, target_matches = _greedily_match(cost, match_th_h)
                assert ((best_matches == 0) | (best_matches == 1)).all()
               
                for best_ind, t_ind in zip(best_matches, target_matches):
                    best_match_skel = halves2check[t_ind]
                    if best_ind == 0:
                        skels_new[:n_segments] = best_match_skel
                    else:
                        skels_new[n_segments-1:] = best_match_skel[::-1]
                
                if target_matches.size:
                    #remove the assigned halves
                    good = np.ones(len(halves2check), np.bool)
                    good[target_matches] = False
                    missing_halves_r[next_frame] = halves2check[good]
            #
            if (points2check.shape[0] > 0) & (np.isnan(skels_new).any()):
                match_th_p = max_frac_dist_seg*skel_length
                
                inds2match, = np.where(np.isnan(skels_new[:, 0]))
                points2match = last_skel[inds2match]
                
                dx = points2match[:, 0][:, None] - points2check['x'][None]
                dy = points2match[:, 1][:, None] - points2check['y'][None]
                cost = np.sqrt((dx*dx) + (dy*dy))
                
                cost = np.sqrt((dx*dx) + (dy*dy))
                best_matches, target_inds = _greedily_match(cost, match_th_p)
                if len(best_matches):
                    matched_points = points2check[best_matches]
                    matched_points = np.array((matched_points['x'], matched_points['y'])).T
                    skels_new[inds2match[target_inds]] = matched_points
                
                    #remove the assigned points
                    good = np.ones(len(points2check), np.bool)
                    good[best_matches] = False
                    missing_points_r[next_frame] = points2check[good]
            
            # plt.figure()
            # plt.plot(last_skel[:, 0], last_skel[:, 1])
            # for h in halves2check:
            #     plt.plot(skels_new[:, 0], skels_new[:, 1])
            
            bad_ = np.isnan(skels_new[:, 0])
            
            
            if np.sum(bad_) > 2:
                break
            else:
                skels_new[bad_] = last_skel[bad_]
                assert not np.isnan(skels_new).any()
                skels.append((next_frame, skels_new))
                
            #%%
                
            
    #%%
    # frame = 4945
    # with tables.File(masked_file, 'r') as fid:
    #     img = fid.get_node('/mask')[frame]
     
    # plt.figure()
    # plt.imshow(img, cmap ='gray')
    
    # skels = [s[1] for ss in matched_skeletons for s in ss if s[0] == frame]
    # for s in skels:
    #     plt.plot(s[:, 0], s[:, 1], '-r')
    
    #%%
    matched_skeletons = sorted(matched_skeletons, key = lambda x : x[0][0])
    initial_skels, final_skels = zip(*[((ii, *x[0]), (ii, *x[-1])) for ii, x in enumerate(matched_skeletons)])
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
    remaining_inds = set(list(range(len(matched_skeletons)))) - used_inds
    
    is_switch = False
    skels_joined = [matched_skeletons[i] for i in remaining_inds]
    #skels_joined = []
    for inds2join in joined_sequences:
        ind, cost = inds2join[0]
        skels = matched_skeletons[ind].copy()
        
        for ind, cost in inds2join[1:]:
            if cost < 0:
                is_switch = ~is_switch 
            
            next_skels = matched_skeletons[ind]
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
    #skels_joined = matched_skeletons
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
            