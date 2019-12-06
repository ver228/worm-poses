#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:01:50 2019

@author: avelinojaver
"""
from pathlib import Path

data_types_dflts = {
    'v1': dict(
            root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190606_174815/',
            flow_args = dict(
                
                skel_size_lims = (60, 200),
             
                n_rois_lims = (1, 1),
                 
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.5),
             
                width2sigma = 0.25,
                min_sigma = 1.,
                
                fold_skeleton = True,
                PAF_seg_dist = 5
                
                )
            ),
    'detection-clusters' : dict(
                root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/',
                
                flow_args = dict(
                 is_clusters_bboxes = True,
                 skel_size_lims = (90, 150),
                 n_rois_lims = (1, 5),
                 n_rois_neg_lims = (1, 5),
                 negative_file = 'negative_from_tierpsy.p.zip',
                 int_aug_offset = (-0.2, 0.2),
                 int_aug_expansion = (0.7, 1.5)
                 
                 )
            ),
    'detection-singles' : dict(
                root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/',
                
                flow_args = dict(
                 is_clusters_bboxes = False,
                 skel_size_lims = (90, 150),
                 n_rois_lims = (1, 5),
                 n_rois_neg_lims = (1, 5),
                 negative_file = 'negative_from_tierpsy.p.zip',
                 int_aug_offset = (-0.2, 0.2),
                 int_aug_expansion = (0.7, 1.5)
                 
                 )
            )
    
    }

