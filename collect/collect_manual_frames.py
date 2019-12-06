#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:51:00 2019

@author: avelinojaver
"""
import pandas as pd
import tables
from pathlib import Path
import tqdm

from smooth_manual_annotations import get_labelled_rois, save_labelled_roi

if __name__ == '__main__':
    _is_debug = False
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/'
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/labelled_rois/manual_from_movies'
    
    
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    
    save_dir.mkdir(parents = True, exist_ok = True)
    
    
    mask_files = [x for x in root_dir.rglob('*.hdf5') if x.parent.name == 'MaskedVideos']
    mask_files = [(ii,x) for ii, x in enumerate(sorted(mask_files, key = lambda x : x.name))]
    
    df = pd.DataFrame([(ii, x.name) for ii, x in mask_files], columns=['movie_id', 'basename'])
    df.to_csv(str(save_dir / 'manual_frames.csv'), index=False)
    
    for movie_id, mask_file in tqdm.tqdm(mask_files):
        
        with tables.File(str(mask_file), 'r') as fid_mask:
            masks = fid_mask.get_node('/mask')
            
            annotations_dir = mask_file.parents[1] / 'Annotations' / mask_file.stem
            annotations_files = sorted(annotations_dir.rglob('*.xml'))
            for annotations_file in annotations_files:
                frame_number = int(annotations_file.stem)
                img_masked = masks[frame_number]
                labelled_rois = get_labelled_rois(annotations_file, img_masked)
                
                for roi_id, roi_data in enumerate(labelled_rois):
                    save_name = save_dir / f'M{movie_id}_frame-{frame_number}-roi-{roi_id}.hdf5'
                    save_labelled_roi(save_name, *roi_data)
                    
                    if _is_debug:
                        
                        roi_masked, roi_full, skels, w, cnt1, cnt2 = roi_data
                        import matplotlib.pylab as plt
                        plt.figure()
                        
                        plt.imshow(roi_masked, cmap = 'gray')
                        #plt.plot(frame_data['coord_x'], frame_data['coord_y'], 'o')
                        
                        for skel, c1, c2 in zip(skels, cnt1, cnt2):
                            plt.plot(skel[:, 0], skel[:, 1])
                            plt.plot(c1[:, 0], c1[:, 1])
                            plt.plot(c2[:, 0], c2[:, 1])
                
                if _is_debug:
                    break