#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:15:45 2020

@author: avelinojaver
"""

from sets2images import get_roi2save


import gzip
from pathlib import Path
import pickle
import cv2
import tqdm

MAX_FILES_PER_FOLDER = 1000

if __name__ == '__main__':
    save_dir_root = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois2check/')
    src_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/')
    
    
    #fname = src_dir / 'manual_validation.p.zip'
    #fname = src_dir / 'from_tierpsy_validation.p.zip'
    
    fnames = src_dir.glob('*.p.zip')
    fnames = [x for x in fnames if not 'negative' in x.name]
    
    for fname in tqdm.tqdm(fnames):
        bn = fname.name[:-6]
        with gzip.GzipFile(fname, 'rb') as fid:
            data = pickle.load(fid)
        
        
        save_dir_images = save_dir_root / 'images' / bn
        save_dir_pickles = save_dir_root / 'pickles' / bn
        
        
        if len(data) <= len(list(save_dir_images.rglob('*.png'))):
            continue
        
        
        part_id = 0
        for ii, out in enumerate(tqdm.tqdm(data, desc = bn)):
            if ii % MAX_FILES_PER_FOLDER == 0:
                part_id += 1
            
            
            roi = out[1] if out[1] is not None else out[0]
            skels = out[3]
            roi2save = get_roi2save(roi, skels)
            
            subdir = f'{bn}_{part_id}'
             
            image_save_name = save_dir_images / subdir / f'{ii}.png'
            image_save_name.parent.mkdir(exist_ok = True, parents = True)
            cv2.imwrite(str(image_save_name), roi2save)
            
            
            pickle_save_name = save_dir_pickles / subdir / f'{ii}.pickle' 
            pickle_save_name.parent.mkdir(exist_ok = True, parents = True)
            
            with open(pickle_save_name, 'wb') as fid:
                pickle.dump(out, fid)
        
        