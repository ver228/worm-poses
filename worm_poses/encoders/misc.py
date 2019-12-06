#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:07:39 2019

@author: avelinojaver
"""

import numpy as np
from scipy.interpolate import interp1d


def remove_duplicated_annotations(skels, widths, cutoffdist = 3.):
    def _calc_dist(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum(axis=1)).mean()
    
    duplicates = []
    for i1, skel1 in enumerate(skels):
        seg_size =  _calc_dist(skel1[1:], skel1[:-1])
        for i2_of, skel2 in enumerate(skels[i1+1:]):
            d1 = _calc_dist(skel1, skel2)
            d2 = _calc_dist(skel1, skel2[::-1])
            d = min(d1, d2)/seg_size
            
            i2 = i2_of + i1 + 1
            if d < cutoffdist:
                duplicates.append(i2)
                
    if duplicates:
        good = np.ones(len(skels), dtype=np.bool)
        good[duplicates] = False
        skels = skels[good]
        
    return skels, widths


def get_best_match(skels_pred, skels_true):
    
    best_matches = []
    skels_a = np.array(skels_true)
    for i1, pred_skel in enumerate(skels_pred):
        assert pred_skel.shape[0] == skels_a.shape[1]
        
        
        ss = pred_skel[None] - skels_a
        s_ori_dist = np.sqrt(ss**2).mean(axis=(1,2))
        
        ss_inv = pred_skel[::-1][None] - skels_a
        s_inv_dist = np.sqrt(ss_inv**2).mean(axis=(1,2))
        s_dist = np.minimum(s_ori_dist, s_inv_dist)
        
        i2 = np.argmin(s_dist)
        best_matches.append((i1, i2, s_dist[i2]))
    
    if best_matches:
        _, selected_true, closest_dist = zip(*best_matches)
    else:
        selected_true, closest_dist = [], []
    
    return selected_true, closest_dist


def resample_curve(curve, resampling_N=49):
    '''Resample curve to have resampling_N equidistant segments'''

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

    fx = interp1d(lengths, curve[:, 0])
    fy = interp1d(lengths, curve[:, 1])

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    # I add the epsilon because otherwise the interpolation will produce nan
    # for zero
    try:
        resampled_curve = np.zeros((resampling_N, 2))
        resampled_curve[:, 0] = fx(subLengths)
        resampled_curve[:, 1] = fy(subLengths)
    except ValueError:
        resampled_curve = np.full((resampling_N, 2), np.nan)

    return resampled_curve