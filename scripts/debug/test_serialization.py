#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:01:12 2019

@author: avelinojaver
"""

import sys

from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import numpy as np
import pickle
import gzip
from worm_poses.flow import remove_duplicated_annotations

class SerializedArrays():
    def __init__(self, array_lists):
        data_ind = 0
        keys = []
        
        for dat in array_lists:
            if dat is not None:
                dtype = dat.dtype
                ndims = dat.ndim
        
        
        for i, dat in enumerate(array_lists):
            if dat is None:
                key = [-1]*(ndims + 2)
            else:
                key = (data_ind, dat.size, *dat.shape)
                data_ind += dat.size
                
            keys.append(key)
        keys = np.array(keys)
        
        if data_ind == 0:
            image_data = np.zeros(0, np.uint8)
        else:
            image_data = np.zeros(data_ind, dtype)
            for key, dat in zip(keys, array_lists):
                if dat is None:
                    continue
                l, r = key[0], key[0] + dat.size
                image_data[l:r] = dat.flatten()
            
        self.keys = keys
        self.data = image_data
        
    def __getitem__(self, ind):
        key = self.keys[ind]
        index = key[0]
        size = key[1]
        shape = key[2:]
        
        if index < 0:
            return None
        
        roi_flatten = self.data[index: index + size]
        roi = roi_flatten.reshape(shape)
        return roi
    
    
    
class SerializedData():
    def __init__(self, data):
        self._size = len(data)
        self.objects = [SerializedArrays(x) for x in zip(*data)]
            
    def __getitem__(self, ind):
        return [x[ind] for x in self.objects]
    
    def __len__(self):
        return self._size
        
            

if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/' 
    set2read = 'validation'
    data_types = ['from_tierpsy', 'manual']
    
    '''I am pooling all the data from varible size images into a single array.
    this is a bit of a hack. It seems like there is a the multiprocessing using in the data loader
    does not like to have hundredths of thousands of small objects.
    https://github.com/pytorch/pytorch/issues/13246#issuecomment-445770039
    Having just a few numpy objects should help.
    '''
    
    
    root_dir = Path(root_dir)
    data = {}
    
    print(f'Loading `{set2read}` from `{root_dir}` ...')
    for data_type in data_types:
        fname = root_dir / f'{data_type}_{set2read}.p.zip'
        
        with gzip.GzipFile(fname, 'rb') as fid:
            data_raw = pickle.load(fid)
        
        data_filtered = []
        for _out in data_raw:
            _out = [np.array(x) for x in _out ]
            roi_mask, roi_full, widths, skels = _out[:4]
            
            if np.isnan(skels).any():
                continue
            if roi_mask.sum() == 0:
                continue
            #there seems to be some skeletons that are an array of duplicated points...
            
            mean_Ls = np.linalg.norm(np.diff(skels, axis=1), axis=2).sum(axis=1)
            if np.any(mean_Ls< 1.):
                continue
            skels, widths = remove_duplicated_annotations(skels, widths)
            
            #if len(skels) < 2: continue
            
            data_filtered.append((roi_mask, roi_full, widths, skels, *_out[4:]))
        data[data_type] = data_filtered
    
    data = {k:SerializedData(v) for k,v in data.items()}
    
    
    #%%
#    roi_masks, roi_fulls, widths, skels, cnts, cnts_bboxes, clusters_bboxes = zip(*data_filtered)
#    
#    roi_masks = SerializedArrays(roi_masks)
#    clusters_bboxes = SerializedArrays(clusters_bboxes)
    
    #%%
    
    
    #roi_mask, roi_full, widths, skels, cnts, cnts_bboxes, clusters_bboxes
    
    
            
    #return data