#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:01:24 2019

@author: avelinojaver
"""

import numpy as np
import ctypes
from multiprocessing import Array
import pickle
import gzip
from pathlib import Path 

def remove_duplicated_annotations(skels, widths, cutoffdist = 3.):
    def _calc_dist(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum(axis=1)).mean()
    
    duplicates = []
    for i1, skel1 in enumerate(skels):
        seg_size =  _calc_dist(skel1[1:], skel1[:-1])
        for i2_of, skel2 in enumerate(skels[i1+1:]):
            d1 = _calc_dist(skel1, skel2)
            d2 = _calc_dist(skel1, skel2[::-1])
            d = min(d1, d2)/seg_size
            
            i2 = i2_of + i1 + 1
            if d < cutoffdist:
                duplicates.append(i2)
                
    if duplicates:
        good = np.ones(len(skels), dtype=np.bool)
        good[duplicates] = False
        skels = skels[good]
        
    return skels, widths

def is_invalid_with_zeros(roi, skels, max_zeros_frac = 0.1):
    skels_i = skels.astype('int')
        
    H,W = roi.shape
    scores = []
    for skel_i in skels_i:
        skel_i[skel_i[:, 0] >= W, 0] = W - 1
        skel_i[skel_i[:, 1] >= H, 1] = H - 1
        ss = roi[skel_i[..., 1], skel_i[..., 0]]
        scores.append((ss==0).mean())
    
    return any([x>max_zeros_frac for x in scores])
        

class SerializedData():
    def __init__(self, data, field_names = [], shared_objects = None, is_read_only = True):
        self.field_names = field_names
        self.serialized_data = [self._serialize_from_list(x, not is_read_only) for x in zip(*data)]
        
        #if data is not None:
            
        #elif shared_objects is not None:
        #    self.serialized_data = [[self._from_share_obj(s) for s in x] for x in shared_objects]
        
        self._size = len(self.serialized_data[0][0])
        
        
        assert self._size == len(data)
        assert len(self.serialized_data) == len(field_names)
        
        
    def __getitem__(self, ind):
        return {k:self._unserialize_data(ind, *x) for k, x in zip(self.field_names, self.serialized_data)}
    
    def __len__(self):
        return self._size
    
    @staticmethod
    def _unserialize_data(ind, keys, array_data):
        key = keys[ind]
        index = key[0]
        size = key[1]
        shape = key[2:]
        
        if index < 0:
            return None
        
        array_flatten = array_data[index: index + size]
        array = array_flatten.reshape(shape)
        return array.copy()
    
    @staticmethod
    def _serialize_from_list(array_lists, write):
        data_ind = 0
        keys = []
        
        
        dtype = np.float
        ndims = 0
        for dat in array_lists:
            if dat is not None:
                dtype = dat.dtype
                ndims = dat.ndim
                break
                
        
        
        for i, dat in enumerate(array_lists):
            if dat is None:
                key = [-1]*(ndims + 2)
            else:
                key = (data_ind, dat.size, *dat.shape)
                data_ind += dat.size
                
            keys.append(key)
        keys = np.array(keys)
        
        if data_ind == 0:
            array_data = np.zeros(0, np.uint8)
        else:
            array_data = np.zeros(data_ind, dtype)
            for key, dat in zip(keys, array_lists):
                if dat is None:
                    continue
                l, r = key[0], key[0] + dat.size
                array_data[l:r] = dat.flatten()
                
        keys.setflags(write=write)
        array_data.setflags(write=write)
        return keys, array_data
    

    #TODO Maybe I should remove the methods below. It was a test to use share memory in order 
    #to improve the performance using mp.set_start_method('fork', force=True)
    @staticmethod
    def _to_share_obj(val):
        dtype  = val.dtype
        
        if dtype == np.int32:
            c_type = ctypes.c_int32
        elif dtype == np.uint8:
            c_type = ctypes.c_uint8
        elif dtype == np.int64:
            c_type = ctypes.c_longlong
        elif dtype == np.float32:
            c_type = ctypes.c_float
        elif dtype == np.float64:
            c_type = ctypes.c_double
        else:
            raise ValueError('dtype `{dtype}` not implemented.')
        
        #https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
        X = Array(c_type, val.size)
        X_np = np.frombuffer(X.get_obj(), dtype = dtype).reshape(val.shape)
        np.copyto(X_np, val)
        
        return X, val.shape, dtype
    
    @staticmethod
    def _from_share_obj(dat):
        X, shape, dtype = dat
        return np.frombuffer(X.get_obj(), dtype).reshape(shape)
    
    def create_share_objs(self):
        return [[self._to_share_obj(s) for s in x] for x in self.serialized_data]
    
def read_data_files( root_dir, 
                     set2read, 
                     data_types = ['from_tierpsy', 'manual'],
                     expected_field_names = ['roi_mask', 'roi_full', 'widths', 'skels', 'contours', 'cnts_bboxes', 'clusters_bboxes'],
                     is_read_only = True
                     ):
        
    
    
    root_dir = Path(root_dir)
    data = {}
    print(f'Loading `{set2read}` from `{root_dir}` ...')
    for data_type in data_types:
        fname = root_dir / f'{data_type}_{set2read}.p.zip'
        
        with gzip.GzipFile(fname, 'rb') as fid:
            data_raw = pickle.load(fid)
        
        data_filtered = []
        for _out in data_raw:
            _out = [x if x is None else np.array(x) for x in _out]
            roi_mask, roi_full, widths, skels = _out[:4]
            
            if roi_mask.sum() == 0:
                continue
            
            #skeletons that are mostly in the black part of the image
            if is_invalid_with_zeros(roi_mask, skels):
                continue
            
            #there are some skeletons that are only one point...
            mean_Ls = np.linalg.norm(np.diff(skels, axis=1), axis=2).sum(axis=1)
            if np.any(mean_Ls< 1.):
                continue
            
             #there seems to be some skeletons that are an array of duplicated points...
            skels, widths = remove_duplicated_annotations(skels, widths)
            
            data_filtered.append((roi_mask, roi_full, widths, skels, *_out[4:]))
        
        data[data_type] = SerializedData(data_filtered, field_names = expected_field_names, is_read_only = is_read_only) 
        
    return data


def read_negative_data( src_files, 
                 is_read_only = True
                 ):
    data_raw = []
    for src_file in src_files:
        src_file = Path(src_file)
        with gzip.GzipFile(src_file, 'rb') as fid:
            data_raw += pickle.load(fid)
    
    data_raw = [(x,) for x in data_raw]
    data = SerializedData(data_raw, field_names = ['image'], is_read_only = is_read_only) 
    return data