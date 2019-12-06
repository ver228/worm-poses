#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:19:28 2019

@author: avelinojaver
"""

import numpy as np
import cv2
import math

#%% SKELETON POINTS BELIEVE MAPS
def _skel2maps(skels, widths, out_shape, width2sigma = 0.25, min_sigma = 1, patch_size = 3):
    
    if width2sigma > 0:
        if isinstance(widths, (float, int)):
            widths = np.full(len(skels), widths)
        sigma = (widths*width2sigma)
        sigma = np.where(sigma < min_sigma, min_sigma, sigma)
        
        
        
        skel_maps = np.zeros((len(skels), *out_shape))
        for i_skel, (sigma_seg, skel_seg) in enumerate(zip(sigma, skels)):
            mu_x, mu_y = skel_seg
            
            
            mu_x_int = int(round(mu_x))
            mu_y_int = int(round(mu_y))
            
            lx = max(mu_x_int - patch_size, 0)
            rx = min(mu_x_int + patch_size + 1, out_shape[1])
            if lx >= rx:
                continue
            
            
            ly = max(mu_y_int - patch_size, 0)
            ry = min(mu_y_int + patch_size + 1, out_shape[0])
            if ly >= ry:
                continue
            
            x_range = np.arange(lx, rx, 1.)
            y_range = np.arange(ly, ry, 1.)
            xx, yy = np.meshgrid(x_range, y_range)
            delta_x = (xx - mu_x)
            delta_y = (yy - mu_y)
            skel_patch = np.exp(-(delta_x**2 + delta_y**2)/(2*sigma_seg**2))
            
            skel_maps[i_skel,  ly:ry, lx:rx] = skel_patch
    else:
        skel_maps = np.zeros((len(skels), *out_shape))
        skels_i = np.floor(skels).astype(np.int)
        
        good = (skels_i[:, 1] < out_shape[0]) &  (skels_i[:, 0] < out_shape[1]) & (skels_i>0).all(axis=1)
        skels_i = skels_i[good]
        for ii, s_ind in enumerate(skels_i):
            skel_maps[ii,  s_ind[1], s_ind[0]] = 1.
    return skel_maps


def get_skeletons_maps(skels, 
                      widths, 
                      roi_shape, 
                      width2sigma = 0.25, 
                      min_sigma = 1, 
                      is_fixed_width = False,
                      fold_skeleton = False
                      ):
    n_segments = skels.shape[1]
    skels_maps = np.zeros((n_segments, *roi_shape), dtype = np.float32)
    #skels_maps = []
    for skel, width in zip(skels, widths):
        
        if is_fixed_width:
            ww = min_sigma
            
        else:
            ww = width
        
        patch_size = max(3, int(np.max(ww)*width2sigma))
        mm = _skel2maps(skel,  
                       ww,  
                       out_shape = roi_shape, 
                       width2sigma = width2sigma,
                       min_sigma = min_sigma,
                       patch_size = patch_size
                       )
        skels_maps = np.maximum(mm, skels_maps)
        
    if fold_skeleton:
        midbody_ind = n_segments//2
        
        skel_maps_folded = np.maximum(skels_maps[:midbody_ind], skels_maps[midbody_ind+1:][::-1])
        skels_maps = np.concatenate((skel_maps_folded, skels_maps[midbody_ind][None]))
        
    return skels_maps



#%% PART AFFINITY MAPS
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
    
    
    dx_w = half_width * np.cos(perp_angles)
    dy_w = half_width * np.sin(perp_angles)
    
    cnt_side1[:, 0] +=  dx_w
    cnt_side1[:, 1] +=  dy_w
    
    cnt_side2[:, 0] -=  dx_w
    cnt_side2[:, 1] -=  dy_w
    
    return cnt_side1, cnt_side2

def _inds2segment(inds, roi_shape, skel, cnt_side1, cnt_side2, fix_orientation = False):
    
    #draw the mask corresponding to a worm segment
    seg_masks = []
    for i1, i2 in inds:
        
        vr = skel[i2] - skel[i1]
        mag = np.linalg.norm(vr)
        if mag == 0:
            #nothing to do here... both segments have the same position
            mm = np.zeros((2, *roi_shape), np.float32)
            seg_masks.append(mm)
        else:
            vr = vr / mag
            
            if fix_orientation:
                ang = np.arctan2(vr[1], vr[0])
                if ang > math.pi:
                    vr *= -1
            
            if i1 > i2:
                i1, i2 = i2, i1
            
            cnt = np.concatenate((cnt_side1[i1:i2+1],cnt_side2[i1:i2+1][::-1]))
            cnt = np.round(cnt).astype(np.int)
            
                
            mask = np.zeros(roi_shape, np.float32)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            seg_mask = vr[:, None, None] * mask[None]     
            
            seg_masks.append(seg_mask)
        
    return seg_masks

def _get_segment_vectors(skel, width, roi_shape, seg_dist = 6):
    
    cnt_side1, cnt_side2 = _skel2cnt(skel, width)
    
    midbody_ind = skel.shape[0]//2
    
    ind = np.arange(0, midbody_ind - seg_dist +1)
    inds_l = np.stack((ind, ind + seg_dist)).T
    
    
    ind = np.arange(skel.shape[0] - seg_dist - 1, midbody_ind - 1, -1)
    inds_r = np.stack((ind + seg_dist, ind)).T
    
    #draw left and right segments. The parts affinity fields come from the worm midbody to each of its extremes
    seg_masks_l = _inds2segment(inds_l, roi_shape, skel, cnt_side1, cnt_side2)
    seg_masks_r = _inds2segment(inds_r, roi_shape, skel, cnt_side1, cnt_side2)
    
    
    inds_c = [(midbody_ind - seg_dist//2, midbody_ind + seg_dist//2)]
    seg_masks_c = _inds2segment(inds_c, roi_shape, skel, cnt_side1, cnt_side2, fix_orientation = True)
    
    seg_masks = np.concatenate((seg_masks_l, seg_masks_c, seg_masks_r[::-1]))

    return seg_masks


def _old_with_overlaps_get_part_affinity_maps(skels, 
                           widths, 
                           roi_shape, 
                           seg_dist,
                           fold_skeleton = False
                           ):
    
    n_segments = skels.shape[1]
    n_affinity_maps = 2*(n_segments//2 - seg_dist + 1) + 1
    
    affinity_maps = np.zeros((n_affinity_maps, 2, *roi_shape), dtype = np.float32)
    overlaped_scores = np.zeros((n_affinity_maps, *roi_shape), dtype = np.float32) 
    
    prev_map = None
    
    #ov = []
    for skel, width in zip(skels, widths):
        next_map = _get_segment_vectors(skel, width, roi_shape, seg_dist = seg_dist)
        affinity_maps += next_map
        
        
        if prev_map is not None:
            bad = (prev_map != 0) & (next_map != 0)
            #affinity_maps[bad] = 0
            affinity_maps[bad] = affinity_maps[bad]/2
            
            overlap_regions = (bad).any(axis=1)
            #get the angle between the segments
            angs = np.arccos((prev_map*next_map).sum(axis=1))
            #shift_angles to be between 1 (parallel) and -1 (perpendicular) np.abs(angs[overlap_regions] - math.pi/2)/math.pi*4 - 1
            
            overlap_score = np.abs(angs[overlap_regions] - math.pi/2)/math.pi*4 - 1
            overlaped_scores[overlap_regions] += overlap_score
        prev_map = next_map
    
    
    if fold_skeleton:
        mid = affinity_maps.shape[0]//2
        p1, p2 = affinity_maps[:mid], affinity_maps[mid+1:][::-1]
        
        #i want to remove any overlap...
        bad = (p1!=0) & (p2!=0)
        affinity_maps_folded = p1 + p2
        #affinity_maps_folded[bad] = 0
        affinity_maps_folded[bad] = affinity_maps_folded[bad]/2
        
        affinity_maps = np.concatenate((affinity_maps_folded, affinity_maps[mid][None]))
        
        overlap_regions = (bad).any(axis=1)
        #get the angle between the segments
        angs = np.arccos((p1*p2).sum(axis=1))
        
        overlaped_scores_folded = overlaped_scores[:mid] + overlaped_scores[mid+1:][::-1]
        #shift_angles to be between 1 (parallel) and -1 (perpendicular)
        overlaped_scores_folded[overlap_regions] = np.abs(angs[overlap_regions] - math.pi/2)/math.pi*4 - 1
        overlaped_scores = np.concatenate((overlaped_scores_folded, overlaped_scores[mid][None]))
     
    
    #plt.figure()
    #plt.imshow(np.abs(overlaped_flags).sum(axis=0))
    #affinity_maps = np.concatenate((affinity_maps, overlaped_scores[:, None]), axis=1)
    
    return affinity_maps

def get_part_affinity_maps(skels, 
                           widths, 
                           roi_shape, 
                           seg_dist,
                           fold_skeleton = False
                           ):
    
    n_segments = skels.shape[1]
    n_affinity_maps = 2*(n_segments//2 - seg_dist + 1) + 1
    
    affinity_maps = np.zeros((n_affinity_maps, 2, *roi_shape), dtype = np.float32)
    
    prev_map = None
    
    for skel, width in zip(skels, widths):
        next_map = _get_segment_vectors(skel, width, roi_shape, seg_dist = seg_dist)
        affinity_maps += next_map
        
        
        if prev_map is not None:
            bad = (prev_map != 0) & (next_map != 0)
            #affinity_maps[bad] = 0
            affinity_maps[bad] = affinity_maps[bad]/2
            
        prev_map = next_map
    
    
    if fold_skeleton:
        mid = affinity_maps.shape[0]//2
        p1, p2 = affinity_maps[:mid], affinity_maps[mid+1:][::-1]
        
        #i want to remove any overlap...
        bad = (p1!=0) & (p2!=0)
        affinity_maps_folded = p1 + p2
        #affinity_maps_folded[bad] = 0
        affinity_maps_folded[bad] = affinity_maps_folded[bad]/2
        
        affinity_maps = np.concatenate((affinity_maps_folded, affinity_maps[mid][None]))
    
    if np.any(np.isnan(affinity_maps)):
        import pdb
        pdb.set_trace()

    
    return affinity_maps
    
