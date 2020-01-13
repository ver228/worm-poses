#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:19:28 2019

@author: avelinojaver
"""

import numpy as np
import cv2
import math

def _skel2cnt(skeleton, w_width):
    '''
    Estimate the worms contours from the skeletons and the widths.
    skeleton -> [n_segments, 2]
    width -> [n_segments]
    '''
    dx = np.diff(skeleton[:, 0])
    dy = np.diff(skeleton[:, 1])

    skel_angles = np.arctan2(dy, dx)
    skel_angles = np.hstack((skel_angles[0], skel_angles))

    #%get the perpendicular angles to define line scans (orientation doesn't
    #%matter here so subtracting pi/2 should always work)
    perp_angles = skel_angles - np.pi / 2
    half_width = w_width/2
    
    cnt_side1 = skeleton.copy()
    cnt_side2 = skeleton.copy()
    
    
    dx_w = np.round(half_width * np.cos(perp_angles)).astype(skeleton.dtype)
    dy_w = np.round(half_width * np.sin(perp_angles)).astype(skeleton.dtype)
    
    cnt_side1[:, 0] +=  dx_w
    cnt_side1[:, 1] +=  dy_w
    
    cnt_side2[:, 0] -=  dx_w
    cnt_side2[:, 1] -=  dy_w
    
    return cnt_side1, cnt_side2

def _inds2segment(inds, 
                  roi_shape, 
                  skel, 
                  cnt_side1, 
                  cnt_side2, 
                  dst_array = None,
                  fix_orientation = False):
    
    #draw the mask corresponding to a worm segment
    
    if dst_array is None:
        dst_array = np.zeros((len(inds), 2, *roi_shape), np.float32)
    else:
        assert dst_array.shape == (len(inds), 2, *roi_shape)
    
    for mask, (i1, i2) in zip(dst_array, inds):
        
        vr = skel[i2] - skel[i1]
        mag = np.linalg.norm(vr)
        
        #if m == 0 both segments have the same position
        if mag != 0:
            vr = vr / mag
            
            if fix_orientation:
                ang = np.arctan2(vr[1], vr[0])
                if ang > math.pi:
                    vr *= -1
            
            if i1 > i2:
                i1, i2 = i2, i1
            
            cnt = np.concatenate((cnt_side1[i1:i2+1],cnt_side2[i1:i2+1][::-1]))
            cnt = np.round(cnt).astype(np.int)
            
            
            for mm, v in zip(mask, vr):
                cv2.drawContours(mm, [cnt], -1, float(v), -1)
                
    return dst_array

def _get_segment_vectors_folded(skel, width, roi_shape, seg_dist = 6, dst_array = None):
    
    cnt_side1, cnt_side2 = _skel2cnt(skel, width)
    
    midbody_ind = skel.shape[0]//2
    
    
    ind = np.arange(0, midbody_ind - seg_dist +1)
    inds_l = np.stack((ind, ind + seg_dist)).T
    
    
    ind = np.arange(skel.shape[0] - seg_dist - 1, midbody_ind - 1, -1)
    inds_r = np.stack((ind + seg_dist, ind)).T[::-1]
    
    ind = max(seg_dist//2, 1)
    inds_c = [(midbody_ind - ind, midbody_ind + ind)]
    
    
    tot_maps = len(inds_l) + len(inds_r) + len(inds_c)
    
    if dst_array is None:
        dst_array = np.zeros((tot_maps, 2, *roi_shape), np.float32)
    
    
    seg_masks_l = dst_array[:len(inds_l)]
    seg_masks_r = dst_array[-len(inds_r):]
    seg_masks_c = dst_array[len(inds_l):len(inds_l)+len(inds_c)]
    
    
    #draw left and right segments. The parts affinity fields come from the worm midbody to each of its extremes
    _inds2segment(inds_l, roi_shape, skel, cnt_side1, cnt_side2, dst_array = seg_masks_l)
    _inds2segment(inds_r, roi_shape, skel, cnt_side1, cnt_side2, dst_array = seg_masks_r)
    _inds2segment(inds_c, roi_shape, skel, cnt_side1, cnt_side2, fix_orientation = True, dst_array = seg_masks_c)
    
    return dst_array

def _get_segment_vectors_straight(skel, width, roi_shape, seg_dist = 6, dst_array = None):
    cnt_side1, cnt_side2 = _skel2cnt(skel, width)
    ind = np.arange(0, skel.shape[0] - seg_dist)
    inds = np.stack((ind, ind + seg_dist)).T
    
    return _inds2segment(inds, roi_shape, skel, cnt_side1, cnt_side2, dst_array = dst_array)

def get_part_affinity_maps(skels, 
                           widths, 
                           roi_shape, 
                           seg_dist,
                           fold_skeleton = False
                           ):
    
    n_segments = skels.shape[1]
    
    
    
    
    if fold_skeleton:
        
        n_affinity_maps = 2*(n_segments//2 - seg_dist + 1) + 1
        affinity_maps = np.zeros((n_affinity_maps, 2, *roi_shape), dtype = np.float32)
        
        for skel, width in zip(skels, widths):
            _get_segment_vectors_folded(skel, width, roi_shape, seg_dist = seg_dist, dst_array = affinity_maps)
    
        mid = affinity_maps.shape[0]//2
        p1, p2 = affinity_maps[:mid], affinity_maps[mid+1:][::-1]
        
        #i want to remove any overlap...
        bad = (p1!=0) & (p2!=0)
        affinity_maps_folded = p1 + p2
        #affinity_maps_folded[bad] = 0
        affinity_maps_folded[bad] = affinity_maps_folded[bad]/2
        
        affinity_maps = np.concatenate((affinity_maps_folded, affinity_maps[mid][None]))
    else:
        
        n_affinity_maps = n_segments - seg_dist
        affinity_maps = np.zeros((n_affinity_maps, 2, *roi_shape), dtype = np.float32)
        
        for skel, width in zip(skels, widths):
            _get_segment_vectors_straight(skel, width, roi_shape, seg_dist = seg_dist, dst_array = affinity_maps)
        
        
    assert not np.any(np.isnan(affinity_maps))
    
    return affinity_maps
    
