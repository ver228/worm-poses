#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 07:20:43 2018

@author: avelinojaver
"""
from pathlib import Path


import shutil

ROI_DATA_COLS = ['movie_id', 'frame', 'worm_index', 'skeleton_id', 'is_skeletonized']

    
#%%
if __name__ == '__main__':
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/manual_annotations/')
    save_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/labelled_rois/')
                    
    
    
    bads_dir = root_dir / 'bad'
    bads_dir.mkdir(exist_ok = True)
    
    for phase_id in ['Phase_7', 'Phase_8', 'Phase_9', 'Phase_10']: #'Phase4', 

        
        annotations_dir = root_dir / phase_id / 'annotations'
        full_dir = root_dir / phase_id / 'full'
        
        annotations_fnames = annotations_dir.glob('*.xml')
        annotations_stems = {x.stem:x for x in annotations_fnames if not x.name.startswith('.')}
    
        images_fnames = full_dir.glob('*.png')
        images_stems = {x.stem:x for x in images_fnames if not x.name.startswith('.')}
        
        bad_keys = set(images_stems.keys()) - set(annotations_stems.keys())
        
        for key in bad_keys:
            src_file = images_stems[key]
            shutil.copy(src_file, bads_dir / src_file.name)
        
        
        
        
        
    
    