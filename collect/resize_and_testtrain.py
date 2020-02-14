#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:19:09 2019

@author: avelinojaver
"""

import tables
from pathlib import Path
import tqdm
import pickle
import gzip
import datetime
import random
import numpy as np
import cv2
#%%
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

#root_dir = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/'
#save_root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/'

root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/labelled_rois/')
save_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/')

#MAX_N_FILES = 110000
MAX_N_FILES = 51000
n_test = 250#500
n_val = 250#500

now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d_%H%M%S')
save_dir = save_root_dir / date_str
save_dir.mkdir(parents = True, exist_ok = True)

for data_type in ['manual-v2']:#[ 'manual', 'from_tierpsy']:
    fnames = (root_dir / data_type).rglob('*.hdf5')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    
    random.shuffle(fnames)
    fnames = fnames[:MAX_N_FILES]
    
    all_data = {'test':[], 'validation':[], 'train':[]}
    
    for ifname, fname in enumerate(tqdm.tqdm(fnames)):
        with tables.File(fname, 'r') as fid:
            
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
            
            #add bounding boxes
            cnt_sides1 = fid.get_node('/contour_side1')[:]
            cnt_sides2 = fid.get_node('/contour_side2')[:]
            
            #test integrity of the data
            if np.isnan(skeletons).any():
                continue
            if roi_mask.sum() == 0:
                continue
            #there seems to be some skeletons that are an array of duplicated points...
            mean_Ls = np.linalg.norm(np.diff(skeletons, axis=1), axis=2).sum(axis=1)
            if np.any(mean_Ls< 1.):
                continue
            
            valid = ~get_duplicates(skeletons)
            skeletons = skeletons[valid].copy()
            widths = widths[valid].copy()
            cnt_sides1 = cnt_sides1[valid].copy()
            cnt_sides2 = cnt_sides2[valid].copy()
            
            cnts = np.concatenate((cnt_sides1, cnt_sides2[:, ::-1]), axis=1).astype(np.int32)
            cnts_bboxes = [(*x.min(axis=0), *x.max(axis=0)) for x in cnts]
            
            roi_bw = np.zeros_like(roi_mask)
            cv2.drawContours(roi_bw, [x[:, None, :] for x in cnts], -1, 255, -1)
            
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_bw, connectivity = 8)
            clusters_bboxes = [(xmin, ymin, xmin + ww, ymin + hh) for xmin, ymin, ww, hh, _ in stats[1:]]
            
            
            if len(cnts_bboxes) != len(skeletons) or len(clusters_bboxes) > len(skeletons):
                import pdb
                pdb.set_trace()
            
            dat = (roi_mask, roi_full, widths, skeletons, cnts, cnts_bboxes, clusters_bboxes)
            
            if ifname < n_test:
                set_type = 'test'
            elif ifname < n_test + n_val:
                set_type = 'validation'
            else:
                set_type = 'train'
            
            all_data[set_type].append(dat)
    
    for set_type, dat in all_data.items():
        save_name = save_dir / f'{data_type}_{set_type}.p.zip'
        with gzip.GzipFile(save_name, 'wb') as fid:
            fid.write(pickle.dumps(dat, 1))
    