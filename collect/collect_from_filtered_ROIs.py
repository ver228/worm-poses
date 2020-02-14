#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:45:36 2020

@author: avelinojaver
"""

from pathlib import Path
import gzip
import pickle
import cv2
from sets2images import get_roi2save

root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/')

src_dir = root_dir / 'rois4training'
save_dir = root_dir / 'rois4training_filtered/'


filtered_file = '/Users/avelinojaver/Desktop/filter_files_120220.txt'
with open(filtered_file) as fid:
    data = fid.read().split('\n')

data = [x.rpartition(' ')[-1] for x in data if x.endswith('.png')]

#%%
valid_indices = {}
for row in data:
    parts = row.split('/')
    set_type = parts[1]
    split_type = parts[2][len(parts[1]) + 1:]
    ind = int(parts[-1][:-4])
    
    key = (set_type, split_type)
    if not key in valid_indices:
        valid_indices[key] = []
        
    valid_indices[key].append(ind)
valid_indices = {k : sorted(x) for k,x in valid_indices.items() }

#%%
is_plot_bad = False

data_filtered = {}
for (set_type, split_type), valid_inds in valid_indices.items():
    
    f2check = f'{set_type}_{split_type}'
    src_file = src_dir / (f2check + '.p.zip')
    with gzip.GzipFile(src_file, 'rb') as fid:
        data = pickle.load(fid)
    
    
    ss = 'manual' if 'manual' in set_type else set_type
    f2save = f'{ss}_{split_type}'
    if not f2save in data_filtered:
        data_filtered[f2save] = []
    data_filtered[f2save] += [data[i] for i in valid_inds]
    
    print(f2check, len(data), len(valid_inds), len(valid_inds)/len(data))
    
    if is_plot_bad:
        check_dir = save_dir / 'bad_images'
        bad_inds = set(range(len(data))) - set(valid_inds)
        for ind in bad_inds:
            out = data[ind]
            roi = out[1] if out[1] is not None else out[0]
            skels = out[3]
            roi2save = get_roi2save(roi, skels)
            image_save_name = check_dir / f2check / f'{ind}.png'
            image_save_name.parent.mkdir(exist_ok = True, parents = True)
            cv2.imwrite(str(image_save_name), roi2save)

#%%
for f2save, data in data_filtered.items():
    file2save = save_dir / (f2save + '.p.zip')
    with gzip.GzipFile(file2save, 'wb') as fid:
        pickle.dump(data, fid)

#%%
# from_NNv1_validation 300 298 0.9933333333333333
# from_NNv1_test 300 296 0.9866666666666667
# from_NNv1_train 9985 9938 0.9952929394091137
# manual_train 22048 21750 0.9864840348330914
# manual_validation 499 490 0.9819639278557114
# manual_test 499 494 0.9899799599198397
# from_tierpsy_train 47278 46917 0.9923643132112188
# from_tierpsy_validation 476 475 0.9978991596638656
# from_tierpsy_test 469 467 0.9957356076759062
# manual-v2_train 8546 7794 0.9120056166627662
# manual-v2_validation 250 211 0.844
# manual-v2_test 250 230 0.92




