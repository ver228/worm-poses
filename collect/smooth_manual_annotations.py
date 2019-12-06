#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:51:00 2019

@author: avelinojaver
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import interp1d

import cv2
import tables
from pathlib import Path
import tqdm

from tierpsy.analysis.ske_create.segWormPython.mainSegworm import resample_curve, smooth_curve, getSkeleton
import re

#%%
def resample_data(skeleton, cnt_side1, cnt_side2, cnt_widths, resampling_N):
    '''I am only resample for the moment'''
    
    
    skeleton, ske_len, cnt_widths = resample_curve(
        skeleton, resampling_N, cnt_widths)
    cnt_side1, _, _ = resample_curve(cnt_side1, resampling_N)
    cnt_side2, _, _ = resample_curve(cnt_side2, resampling_N)
    
    # resample data
    skeleton = smooth_curve(skeleton)
    cnt_widths = smooth_curve(cnt_widths)
    cnt_side1 = smooth_curve(cnt_side1)
    cnt_side2 = smooth_curve(cnt_side2)


    return skeleton, cnt_side1, cnt_side2, cnt_widths

def resample2pix(curve):
    
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)

    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]
    

    fx = interp1d(lengths, curve[:, 0], kind='linear')
    fy = interp1d(lengths, curve[:, 1], kind='linear')

    resampling_N = int(lengths[-1].round()) 

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    resampled_curve = np.zeros((resampling_N, 2))
    resampled_curve[:, 0] = fx(subLengths)
    resampled_curve[:, 1] = fy(subLengths)
        
    resampled_curve = np.round(resampled_curve)
    
    return resampled_curve

def get_skeleton_from_sides(side1, side2):
    
    cnt_side1, cnt_side2 = side1, side2
    
    ht_dist = np.sum((cnt_side2[0][None] - cnt_side1[[0, -1]])**2, axis=1)
    if ht_dist[0] > ht_dist[1]:
        cnt_side1 = cnt_side1[::-1]
    
    
    
    head_coord = (cnt_side1[0] + cnt_side2[0])/2
    tail_coord = (cnt_side1[-1] + cnt_side2[-1])/2
    
    
    cnt_side1 = np.concatenate((head_coord[None], cnt_side1[1:-1], tail_coord[None]), axis=0)
    cnt_side2 = np.concatenate((head_coord[None], cnt_side2[1:-1], tail_coord[None]), axis=0)
    
    cnt_side1 = resample2pix(cnt_side1)
    cnt_side2 = resample2pix(cnt_side2)
    
    
    ii = 1
    cc = np.concatenate((cnt_side1[ii:ii+3], cnt_side1[ii][None]))
    #contours must be in clockwise direction otherwise the angles algorithm will give a bad result
    
    # make sure the contours are in the counter-clockwise direction
    # x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    signed_area = np.sum(
        cc[:-1, 0] * cc[1:, 1] - cc[1:, 0] * cc[:-1, 1]) / 2
    if signed_area > 0:
        cnt_side1, cnt_side2 = cnt_side2, cnt_side1
    
    
    contour = np.concatenate((cnt_side1, cnt_side2[::-1]), axis=0)
    #contour = np.ascontiguousarray(contour[::-1])
    
    skeleton, _, cnt_side1, cnt_side2,  cnt_widths, _ = getSkeleton(contour)

    # resample data
    skeleton = smooth_curve(skeleton)
    cnt_widths = smooth_curve(cnt_widths)
    cnt_side1 = smooth_curve(cnt_side1)
    cnt_side2 = smooth_curve(cnt_side2)
    
    return skeleton, cnt_widths, cnt_side1, cnt_side2

def read_annotations_file(annotations_file):
    only_numeric = re.compile(r'[^\d]+')
    with open(annotations_file, 'r') as fid:
        xml_str = fid.read()
    
    if  xml_str.startswith('<Worm>'):
        xml_str = '<Data>' + xml_str + '</Data>'
    
    root = ET.fromstring(xml_str)
    
    all_sides = []
    for worm in root:
        #there must be 2 sides
        worm = [x for x in worm if x.tag == 'Side']
        assert len(worm) == 2

        sides = []
        for side in worm:
            points = {'x':[], 'y':[]}
            for p in side:
                for c in p:
                    
                    txt = only_numeric.sub('', c.text)
                    
                    points[c.tag].append(int(txt))
                
            #ss = np.array((points['x'], points['y']))
            ss = np.array((points['y'], points['x'])).T
            sides.append(ss)
        
        #ignore sides that have only one element
        if any([x.shape[0] <= 5 for x in sides]):
            continue
        
        all_sides.append(sides)
    return all_sides


def process_annotations_file(annotations_file, roi_pad = 5):
    frame_annotations = read_annotations_file(annotations_file)
    
    #smooth, resample and calculate the skeleton
    dat = [get_skeleton_from_sides(*x) for x in frame_annotations]
    #remove annotations where the skeletonization failed
    dat = [x for x in dat if x[0].size > 0]
    if not dat:
        return None
    
    
    #repack
    skeletons, cnt_widths, cnt_sides1, cnt_sides2 =  map(np.array, zip(*dat))
    
    return skeletons, cnt_widths, cnt_sides1, cnt_sides2

def get_labelled_masked_rois(annotations_file, img_masked, img_full = None):
    
    _out = process_annotations_file(annotations_file)
    if _out is None:
        return []
    
    skeletons, cnt_widths, cnt_sides1, cnt_sides2 = _out
    cnts = np.concatenate((cnt_sides1, cnt_sides2[:, ::-1]), axis=1).astype(np.int32)
    cnts = [x[:, None, :] for x in cnts]
    
    
    roi_mask = np.zeros_like(img_masked)
    
    cv2.drawContours(roi_mask, cnts, -1, 255, -1)
    roi_mask = cv2.dilate(roi_mask, kernel = np.ones((3,3)), iterations = 10)
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, connectivity = 4)
    #columns = ['left', 'top', 'width', 'height', 'area'] 
    
    inds = skeletons[:, 25].round().astype(np.int)
    labs = labels[inds[:, 1], inds[:, 0]]
    if not np.all(labs > 0):
        import pdb
        pdb.set_trace()
        raise ValueError('Something went wrong')
    
    #I want to ignore the first index that corresponds to the bgnd
    stats = stats[1:]
    centroids = centroids[1:]
    
    labelled_rois = []
    for iroi, stat in enumerate(stats):
        left, top, width, height, area = stat
        roi_masked = img_masked[top:top + height, left:left + width]
        
        if img_full is not None:
            roi_full = img_full[top:top + height, left:left + width]
        else:
            roi_full = None
        
        _valid = labs == iroi + 1
        
        cc = np.array((left, top))[None, None]
        
        skels = (skeletons[_valid] - cc).astype(np.float32)
        cnt1 = (cnt_sides1[_valid] - cc).astype(np.float32)
        cnt2 = (cnt_sides2[_valid] - cc).astype(np.float32)
        
        w = cnt_widths[_valid].astype(np.float32)
        
        labelled_rois.append((roi_masked, roi_full, skels, w, cnt1, cnt2))
    
    return labelled_rois
#%%
def get_labelled_full_rois(annotations_file, roi_full):
    
    _out = process_annotations_file(annotations_file)
    if _out is None:
        return []
    
    skeletons, cnt_widths, cnt_sides1, cnt_sides2 = _out
    cnts = np.concatenate((cnt_sides1, cnt_sides2[:, ::-1]), axis=1).astype(np.int32)
    cnts = [x[:, None, :] for x in cnts]
    
    labelled_rois = [None, roi_full, skeletons, cnt_widths, cnt_sides1, cnt_sides2]
    
    return labelled_rois

def get_labelled_rois(annotations_file, img_masked, img_full):
    if img_masked is None:
        labelled_rois = get_labelled_full_rois(annotations_file, img_full)
    else:
        labelled_rois = get_labelled_masked_rois(annotations_file, img_masked, img_full)
        
    return labelled_rois

#%%
def save_labelled_roi(save_name, roi_masked, roi_full, skels, w, cnt1, cnt2):
    TABLE_FILTERS = tables.Filters(
                    complevel=5,
                    complib='blosc',
                    shuffle=True,
                    fletcher32=True)
    
    with tables.File(str(save_name), 'w') as fid:
        
        
        fid.create_carray('/', 
                    'skeletons',
                    obj = skels,
                    filters = TABLE_FILTERS
                    )
        
        fid.create_carray('/', 
                    'widths',
                    obj = w,
                    filters = TABLE_FILTERS
                    )
        
        
        fid.create_carray('/', 
                    'contour_side1',
                    obj = cnt1,
                    filters = TABLE_FILTERS
                    )
        
        fid.create_carray('/', 
                    'contour_side2',
                    obj = cnt2,
                    filters = TABLE_FILTERS
                    )
        
        if roi_full is not None:
            fid.create_carray('/', 
                    'roi_mask',
                    obj = roi_masked,
                    filters = TABLE_FILTERS
                    )
        
        if roi_full is not None:
            fid.create_carray('/', 
                        'roi_full',
                        obj = roi_full,
                        filters = TABLE_FILTERS
                        )


    
    