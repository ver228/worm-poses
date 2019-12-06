#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:09:32 2019

@author: avelinojaver
"""
from pathlib import Path
import tables
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib import patches

def get_duplicates(skels, cutoffdist = 3.):
    def _calc_dist(x1, x2): 
        return np.sqrt(((x1 - x2)**2).sum(axis=1)).mean()
    
    duplicates_inds = []
    for i1, skel1 in enumerate(skels):
        seg_size =  _calc_dist(skel1[1:], skel1[:-1])
        for i2_of, skel2 in enumerate(skels[i1+1:]):
            d1 = _calc_dist(skel1, skel2)
            d2 = _calc_dist(skel1, skel2[::-1])
            d = min(d1, d2)/seg_size
            
            i2 = i2_of + i1 + 1
            if d < cutoffdist:
                duplicates_inds.append(i2)
    
    is_duplicates = np.zeros(len(skels), dtype=np.bool)
    if duplicates_inds:
        is_duplicates[duplicates_inds] = True
        
    return is_duplicates

if __name__ == '__main__':
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_movies/M17_frame-14771-roi-0.hdf5'
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I9970-R0_movie-16146_worm-1_frame-12000.hdf5'
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I6305-R0_movie-4428_worm-7301_frame-65000.hdf5'	
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I998-R0_movie-201_worm-3388_frame-20000.hdf5'	
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase4/I2325-R0_movie-1038_worm-1_frame-010000.hdf5'
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I631-R0_movie-4480_worm-7212_frame-40000.hdf5'
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I9898-R0_movie-14601_worm-1_frame-12000.hdf5'
    #src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I6114-R0_movie-4931_worm-231_frame-55000.hdf5'
    src_file = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual/manual_from_images/Phase5/I10902-R0_movie-4812_worm-1451_frame-10000.hdf5'
    
    with tables.File(src_file, 'r') as fid:
        if '/roi_full' in fid:
            roi_full = fid.get_node('/roi_full')[:]
        else:
            roi_full = None
        
        if '/roi_mask' in fid:
            roi_mask = fid.get_node('/roi_mask')[:]
        else:
            roi_mask = None
            
        skeletons = fid.get_node('/skeletons')[:]
        widths = fid.get_node('/widths')[:]
        
        cnt_sides1 = fid.get_node('/contour_side1')[:]
        cnt_sides2 = fid.get_node('/contour_side2')[:]
        cnts = np.concatenate((cnt_sides1, cnt_sides2[:, ::-1]), axis=1).astype(np.int32)
        
        
        valid = ~get_duplicates(skeletons)
        skeletons = skeletons[valid].copy()
        widths = widths[valid].copy()
        cnts = cnts[valid].copy()
        
        
        cnts_bboxes = [(*x.min(axis=0), *x.max(axis=0)) for x in cnts]
        
        roi_bw = np.zeros_like(roi_mask)
        cv2.drawContours(roi_bw, [x[:, None, :] for x in cnts], -1, 255, -1)
        #roi_bw = cv2.dilate(roi_bw, kernel = np.ones((3,3)))
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_bw, connectivity = 8)
        clusters_bboxes = [(xmin, ymin, xmin + ww, ymin + hh) for xmin, ymin, ww, hh, _ in stats[1:]]
        
        
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(roi_mask)
        axs[1].imshow(roi_bw)
        
        for ss in stats[1:]:
            xmin, ymin, ww, hh, _ = ss
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            axs[0].add_patch(rect)
            
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            axs[1].add_patch(rect)
            
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(roi_mask)
        axs[1].imshow(roi_bw)
        
        for xmin, ymin, xmax, ymax in cnts_bboxes:
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            axs[0].add_patch(rect)
            
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            axs[1].add_patch(rect)
    