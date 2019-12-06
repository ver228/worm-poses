#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:43:37 2019

@author: avelinojaver
"""


import numpy as np
import cv2
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

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


def extrack_keypoints(belive_maps, threshold_relative = 0.25, threshold_abs = .25):
    #get keypoints from local maxima
    keypoints = []
    for belive_map in belive_maps:
        res = cv2_peak_local_max(belive_map, threshold_relative, threshold_abs)
        #map_scores = belive_map[map_coords[:, 0], map_coords[:, 1]]
        #res = np.concatenate((map_coords, map_scores[:, None]), axis=1)
        keypoints.append(res)
        
    #the edges are head/tail to midbody. I want to invert them...
    keypoints = keypoints[::-1]
    
    return keypoints


def join_skeletons_simple(keypoints, max_edge_dist = 10):
    n_expected_halfs = np.median([len(x) for x in keypoints[1:]]) # I am skiping the midbody that should have half of the points
    if n_expected_halfs <= 1:
        #no valid keypoints. nothing to do here...
        return []
    
    n_expected_halfs = math.floor(n_expected_halfs)
    n_expected_halfs = n_expected_halfs if n_expected_halfs%2 == 0 else n_expected_halfs + 1
    
    if len(keypoints[0]) == n_expected_halfs//2:
        #if the mid
        mid_bodies = [x for x in keypoints[0]]
        seeds = [[x] for x in mid_bodies + mid_bodies]
        ini_seg = 1
    else:
        for ii in range(1, len(keypoints)//4):
            if len(keypoints[ii]) == n_expected_halfs:
                seeds = [[x] for x in keypoints[ii]]
                ini_seg = ii + 1
                break
        else:
            #I cannot find a sensible number of seeds so i am exiting
            return []
    
    for next_keypoints in keypoints[ini_seg:]:
        if len(next_keypoints) == 0:
            continue
        
        cost_matrix = []
        for seed in seeds:
            ss = seed[-1]
            
            seed_coords = ss[:2]
            
            delta = seed_coords[None] - next_keypoints[:, :2]
            dist = np.sqrt((delta**2).sum(axis=1))
            cost_matrix.append(dist)
        cost_matrix = np.array(cost_matrix)
        
        #I am adding a second column that cost the double just to allow for cases 
        #where it might be convenient to repeat the lower dimenssion
        cost_matrix = np.concatenate((cost_matrix, cost_matrix*10), axis=0)
        cost_matrix[cost_matrix>max_edge_dist] = max_edge_dist
        col_ind, row_ind = linear_sum_assignment(cost_matrix)
        
        good = cost_matrix[col_ind, row_ind] < max_edge_dist
        col_ind, row_ind = col_ind[good], row_ind[good]
        
        for c, r in zip(col_ind, row_ind):
            if c < len(seeds) and r < len(next_keypoints):
                seeds[c].append(next_keypoints[r])
       
    skel_halves = [np.array(x) for x in seeds]
    
    # join halves
    mid_coords = [x[0] for x in skel_halves]
    
    n_candidates = len(mid_coords)
    
    if n_candidates == 2:
        matches = [(0, 1)] #there is only one pair, nothing to do here
    else:
        dist = pdist(mid_coords)
        dist = squareform(dist)
        np.fill_diagonal(dist, np.max(dist))
        col_ind, row_ind = linear_sum_assignment(dist)
        matches = set([(c,r) if c > r else (r, c) for c,r in zip(col_ind, row_ind)])
    
    skels_pred = []
    for i1, i2 in matches:
        h1, h2 = skel_halves[i1][::-1, :2], skel_halves[i2][:, :2]
        skel = np.concatenate((h1[:, ::-1], h2[:, ::-1]))
        skels_pred.append(skel)
    
    return skels_pred