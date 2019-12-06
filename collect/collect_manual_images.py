#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:51:00 2019

@author: avelinojaver
"""

import cv2
from pathlib import Path
import tqdm

from smooth_manual_annotations import get_labelled_rois, save_labelled_roi
#%%
def groupbybasename(fnames):
    grouped_files = {}
    for fname in fnames:
        bn = fname.stem
        if not bn in grouped_files:
            grouped_files[bn] = []
        grouped_files[bn].append(fname)
    
    valid_files = {k : v[0] for k,v in grouped_files.items() if len(v) == 1}
    replicated_files = {k : v for k,v in grouped_files.items() if len(v) > 1}
    
    return valid_files, replicated_files

if __name__ == '__main__':
    _is_debug = False
            
    root_dir = Path.home() / 'workspace/WormData/worm-poses/raw/rois/'
    save_dir = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/manual_from_images'
    
    for phase_id in ['Phase5']: #'Phase4', 

        
        annotations_dir = root_dir / phase_id / 'annotations'
        full_dir = root_dir / phase_id / 'full'
        mask_dir = root_dir / phase_id / 'mask'
        
        fnames = annotations_dir.glob('*.xml')
        fnames = [x for x in fnames if not x.name.startswith('.')]
        for img_id, annotations_file in enumerate(tqdm.tqdm(fnames)):
            #save_name = save_dir / phase_id/ f'I{img_id}-R0_{annotations_file.stem}.hdf5'
            #if save_name.exists():
            #    continue

            img_name = annotations_file.stem + '.png'
            
            full_fname = full_dir / img_name
            mask_fname = mask_dir / img_name
            
            assert mask_fname.exists or full_fname.exists()
            
            img_full = cv2.imread(str(full_fname), -1)
            if img_full is not None:
                img_full = img_full.T.copy()
            
            img_masked = cv2.imread(str(mask_fname), -1)
            if img_masked is not None:
                img_masked = img_masked.T.copy()
            
            
            labelled_rois = get_labelled_rois(annotations_file, img_masked, img_full)
            for roi_id, roi_data in enumerate(labelled_rois):
                save_name = save_dir / phase_id/ f'I{img_id}-R{roi_id}_{annotations_file.stem}.hdf5'
                save_name.parent.mkdir(exist_ok = True, parents = True)
                
                
                save_labelled_roi(save_name, *roi_data)
                
                if _is_debug:
                    
                    roi_masked, roi_full, skels, w, cnt1, cnt2 = roi_data
                    import matplotlib.pylab as plt
                    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
                    
                    axs[0].imshow(roi_masked, cmap = 'gray')
                    axs[1].imshow(roi_full, cmap = 'gray')
                    
                    for ax in axs:
                        for skel, c1, c2 in zip(skels, cnt1, cnt2):
                            ax.plot(skel[:, 0], skel[:, 1])
                            ax.plot(c1[:, 0], c1[:, 1])
                            ax.plot(c2[:, 0], c2[:, 1])
                    plt.show()
                    
            if _is_debug and img_id > 5:
                break