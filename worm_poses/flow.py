#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:11:22 2018

@author: avelinojaver
"""

from .encoders import remove_duplicated_annotations, get_skeletons_maps, get_part_affinity_maps

import gzip
import pickle

import numpy as np
import random
import cv2

from pathlib import Path
from torch.utils.data import Dataset
import math

import ctypes
import torch
from torch.utils.data.dataloader import default_collate
from multiprocessing import Array

#%%
def collate_with_skels(batch):
    skels = [x[1][-1] for x in batch]
    
    batch_simple = [(x[0], x[1][:-1]) for x in batch]
    X, Y = default_collate(batch_simple)
    return X, (*Y, skels)
    
def collate_simple(batch):
    return tuple(map(list, zip(*batch)))

    
class SerializedData():
    def __init__(self, data = None, shared_objects = None):
        if data is not None:
            self._size = len(data)
            self.serialized_data = [self._serialize_from_list(x) for x in zip(*data)]
        elif shared_objects is not None:
            self.serialized_data = [[self._from_share_obj(s) for s in x] for x in shared_objects]
            self._size = len(self.serialized_data[0][0])
        
    def __getitem__(self, ind):
        return [self._unserialize_data(ind, *x) for x in self.serialized_data]
    
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
        
        roi_flatten = array_data[index: index + size]
        roi = roi_flatten.reshape(shape)
        return roi 
    @staticmethod
    def _serialize_from_list(array_lists):
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
            array_data = np.zeros(0, np.uint8)
        else:
            array_data = np.zeros(data_ind, dtype)
            for key, dat in zip(keys, array_lists):
                if dat is None:
                    continue
                l, r = key[0], key[0] + dat.size
                array_data[l:r] = dat.flatten()
        return keys, array_data
    

    # I am leaving the methods below for the moment, but it is likely they should be removed.
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
        
    
def read_data_files(
                     root_dir, 
                     set2read, 
                     data_types = ['from_tierpsy', 'manual'],
                     is_serialized = True
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
    
    if is_serialized:
        data = {k:SerializedData(v) for k,v in data.items()}
    return data
    


class SkelMapsSimpleFlow(Dataset):
    _valid_headtail = ['from_tierpsy']
    
    def __init__(self, 
                 data = None,
                 root_dir = None, 
                 set2read = 'validation',
                 data_types = ['from_tierpsy', 'manual'],
                 scale_int = (0, 255),
                 roi_size = 128,
                 fold_skeleton = True,
                 return_affinity_maps = True,
                 PAF_seg_dist = 5,
                 is_fixed_width = False,
                 n_segments = 49,
                 width2sigma = 0.25,
                 min_sigma = 2.,
                 return_raw_skels = False
                 ):
        
        self.roi_size = roi_size
        
        self.scale_int = scale_int
        
        self.return_affinity_maps = return_affinity_maps
        self.PAF_seg_dist = PAF_seg_dist
        self.fold_skeleton = fold_skeleton
        
        self.n_segments = n_segments
        self.n_affinity_maps = 2*(self.n_segments//2 - self.PAF_seg_dist + 1) + 1
        self.n_skel_maps_out = self.n_segments // 2 + 1 if self.fold_skeleton else self.n_segments
        self.n_affinity_maps_out = self.n_affinity_maps // 2 + 1 if self.fold_skeleton else self.n_affinity_maps
        
        self.is_fixed_width = is_fixed_width
        self.width2sigma = width2sigma
        self.min_sigma = min_sigma
        
        self.return_raw_skels = return_raw_skels
        
        if data is None:
            self.root_dir = Path(root_dir)
            self.set2read = set2read
            self.data_types = data_types
            self.data = read_data_files(root_dir, set2read, data_types)
        else:
            self.data = data
            #self.data = {k : SerializedData(shared_objects=x) for k,x in data.items()}
        self.data_types = list(self.data.keys())
        self.data_inds = [(k, ii) for k,val in self.data.items() for ii in range(len(val))]
        
        
        #del self.data
    
    def __getitem__(self, ind):
        k, ii = self.data_inds[ind]
        roi_mask, roi_full, widths, skels =  self.data[k][ii][:4]
        roi_mask, skels, widths = roi_mask.copy(), skels.copy(), widths.copy() #I want to modify this without altering the presaved data
        
        if not k in self._valid_headtail: #I am assuming 'from_tierpsy' has the correct head/tail orientation
            if random.random() > 0.5:
                skels = skels[:, ::-1] #switch head/tail as a form of augmentation
        
        roi = roi_mask if roi_full is None else roi_full
        roi = (roi - self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
        
        l_max = max(roi.shape)
        if l_max > self.roi_size:
            scale_f = self.roi_size/l_max
            roi = cv2.resize(roi, dsize = (0,0), fx = scale_f, fy = scale_f)
            assert max(roi.shape) <= self.roi_size
            
            skels *= scale_f
            widths *= scale_f
            
        dd = [self.roi_size - x for x in roi.shape]
        _pad = [(math.floor(x/2), math.ceil(x/2)) for x in dd]
        roi = np.pad(roi, _pad, 'constant', constant_values = 0)
        roi = roi.astype(np.float32)
        
        skels[..., 0] += _pad[1][0]
        skels[..., 1] += _pad[0][0]
        
        
        _out = self._prepare_skels(skels, widths, roi.shape)
        return roi[None], _out
    
    def _prepare_skels(self, _skels, _widths, roi_shape):
        skel_maps = get_skeletons_maps(_skels, 
                                      _widths, 
                                      roi_shape, 
                                      width2sigma = self.width2sigma, 
                                      min_sigma = self.min_sigma, 
                                      is_fixed_width = self.is_fixed_width,
                                      fold_skeleton = self.fold_skeleton
                                      )
        skel_maps = skel_maps.astype(np.float32)
        
       
        if self.return_affinity_maps:
            affinity_maps = get_part_affinity_maps(_skels, 
                                          _widths, 
                                          roi_shape, 
                                          self.PAF_seg_dist,
                                          fold_skeleton = self.fold_skeleton
                                          )
            affinity_maps = affinity_maps.astype(np.float32)
            
            out = skel_maps, affinity_maps
        else:
            out = skel_maps,
        
        if self.return_raw_skels:
            out = [*out, _skels]
    
        return out
    
    
    def __len__(self):
        return len(self.data_inds)
#%%
class SkelMapsRandomFlow(SkelMapsSimpleFlow):
    def __init__(self, 
                 *args,
                 epoch_size = 10000,
                 skel_size_lims = (120, 300),
                 
                 n_rois_lims = (1, 1),
                 
                 int_aug_offset = (-0.2, 0.2),
                 int_aug_expansion = (0.7, 1.5),
                 
                 fold_skeleton = True,
                 
                 **argkws
                 ):
        
        super().__init__(*args, **argkws)
        
        self.epoch_size = epoch_size
        
        self.skel_size_lims = skel_size_lims
        self.skel_size_mu = np.mean(skel_size_lims)
        self.n_rois_lims = n_rois_lims
        
        self.fold_skeleton = fold_skeleton
        
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        
    
    def __getitem__(self, ind):
        return self._build_rand_img()
       
        
    
    def _build_rand_img(self):
        img = np.zeros((self.roi_size, self.roi_size), np.float32)
        
        skel_map = np.zeros((self.n_skel_maps_out, self.roi_size, self.roi_size), np.float32)
        if self.return_affinity_maps:
            affinity_maps = np.zeros((self.n_affinity_maps_out, 2, self.roi_size, self.roi_size), np.float32)
        
        n_rois = random.randint(*self.n_rois_lims)
        
        while n_rois > 0:
            
            roi, roi_out = self._read_roi_rand()
            
            roi_skel_map = roi_out[0]
            if self.return_affinity_maps:
                roi_affinity_map = roi_out[1]
            
            n_rois -= 1
            
            for _ in range(5):
                #try to position the current raw times
                ii = random.randint(0, self.roi_size - roi.shape[0])
                jj = random.randint(0, self.roi_size - roi.shape[1])
                roi_in_img = img[ii: ii + roi.shape[0], jj:jj + roi.shape[1]].view()
                
                if np.any(roi_in_img > 0):
                    continue
                
                roi_in_img += roi
                skel_map[:, ii: ii + roi.shape[0], jj:jj + roi.shape[1]] += roi_skel_map
                
                if self.return_affinity_maps:
                    affinity_maps[..., ii: ii + roi.shape[0], jj:jj + roi.shape[1]] += roi_affinity_map
                
                break
        
        
        
        if np.isnan(img).any() or np.isnan(skel_map).any():
            raise ValueError
            
        if self.return_affinity_maps and np.isnan(affinity_maps).any():
            
            raise ValueError('Critical error. The affinity map contains nan-values.')
            
        if self.return_affinity_maps:
            _out = skel_map, affinity_maps
        else:
            _out = skel_map
            
        
        return img[None], _out
    
    def _read_roi_rand(self):
        set_k = random.choice(self.data_types)
        _dat = random.choice(self.data[set_k])
        roi_mask, roi_full, widths, skels = _dat[:4]
        
        
        assert not (roi_mask is None and roi_full is None)
        if (roi_full is not None) and (random.random() > 0.5):
            roi = roi_full.copy()
        else:
            roi = roi_mask.copy()
        skels, widths = skels.copy(), widths.copy()
        
        #randomly scale to match image or the whole range
        roi = (roi - self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
        
        
        params = self._transforms_params(roi, skels)
        
        roi = self._transforms_roi(roi, **params)
        
        skels, widths = self._transforms_skels(skels, widths, **params)
        out = self._prepare_skels(skels, widths, roi.shape)
        
        return roi, out
    
    def _transforms_roi(self, 
                        roi, 
                        zoom_factor, 
                        is_v_flip,
                        is_h_flip,
                        is_transpose,
                        is_black_patch,
                        int_offset,
                        int_expansion,
                        **argkws
                        ):
        roi = cv2.resize(roi, dsize = (0,0), fx = zoom_factor, fy = zoom_factor)#, cv2.INTER_LINEAR
        if is_v_flip:
            roi = roi[::-1]
            
        if is_h_flip:
            roi = roi[:, ::-1]
        
        if is_transpose:
            roi = roi.T
            
        if is_black_patch:
            roi = self._add_black_patch(roi)
    
        if int_offset is not None:
            roi += int_offset
            
        if int_expansion is not None:
            roi *= int_expansion
        
        return roi
    
    def _transforms_skels(self,
                           skels, 
                           widths,
                            roi_shape,                           
                            zoom_factor, 
                            is_v_flip,
                            is_h_flip,
                            is_transpose,
                            **argkws
                            ):
        
        skels *= zoom_factor
        widths *= zoom_factor
        
        roi_shape_z = [x*zoom_factor for x in roi_shape]
        if is_v_flip:
            skels[..., 1] = roi_shape_z[0] - skels[..., 1]
            
        if is_h_flip:
            skels[..., 0] = roi_shape_z[1] - skels[..., 0]
        
        if is_transpose:
            skels = skels[..., ::-1]
        return skels, widths
    
    def _add_black_patch(self, roi):
        
        blank_patch_size = int(min(roi.shape)/8)
        
        if blank_patch_size <= 1:
            return roi
        
        def _gauss_random(_size):
            b = random.gauss(0, 0.4)/2 + 0.5
            #clip
            b = max(0, min(1, b))
            #scale
            b *= _size-blank_patch_size
            return int(round(b))
            
        ii = _gauss_random(roi.shape[0])
        jj = _gauss_random(roi.shape[1])
        
        roi[ii:ii+blank_patch_size, jj:jj+blank_patch_size] = random.random()
        return roi
    
    def _get_zoom_factor(self, roi, skels = None):
        
        if skels is not None:
            mean_L = np.linalg.norm(np.diff(skels, axis=1), axis=2).sum(axis=1).mean()
            new_L = random.randint(*self.skel_size_lims)
                
            if mean_L > 0.:
                z = new_L/mean_L
            else:
                z = 1.
        else:
            _zoom_range = (0.8, 1.2) #I probably should move this to the args list
            z = random.uniform(*_zoom_range)
    
        new_shape = [int(round(x*z)) for x in  roi.shape]
        new_shape = tuple([min(self.roi_size, x) for x in new_shape])
        
        fx = new_shape[1]/roi.shape[1]
        fy = new_shape[0]/roi.shape[0]
        z_corr = min(fx, fy)
        
        return z_corr
    
    
    def _transforms_params(self, roi, skels = None):
        
        
        if self.int_aug_offset is not None and random.random() > 0.5:
            int_offset = random.uniform(*self.int_aug_offset)
        else:
            int_offset = None
        
        
        if self.int_aug_expansion is not None and random.random() > 0.5:
            int_expansion = random.uniform(*self.int_aug_expansion)
        else:
            int_expansion = None
        
        params = dict(
                zoom_factor = self._get_zoom_factor(roi, skels),
                is_v_flip = True,#random.random() > 0.5,
                is_h_flip = False,#random.random() > 0.5,
                is_transpose = True,#random.random() > 0.5,
                is_black_patch = random.random() > 0.5,
                int_offset = int_offset,
                int_expansion = int_expansion,
                roi_shape = roi.shape
                )
        
        return params

    def __len__(self):
        return self.epoch_size


class RandomFlowDetection(SkelMapsRandomFlow):
    def __init__(self, 
                 is_clusters_bboxes = True, 
                 negative_file = None,
                 n_rois_neg_lims = (2, 5),
                 return_skeletons = True,
                 **argkws
                 ):
        super().__init__( **argkws)
        
        
        self.n_rois_neg_lims = n_rois_neg_lims
        self.is_clusters_bboxes = is_clusters_bboxes
        
        if negative_file is None:
            self.negative_data = None
        else:
            
            negative_file = Path(negative_file) 
            if not negative_file.exists():
                negative_file = Path(self.root_dir) / negative_file
    
            
            with gzip.GzipFile(negative_file, 'rb') as fid:
                self.negative_data = pickle.load(fid)
        
        
        
        
    def _transforms_bbox(self,
                           bboxes,
                            roi_shape,                           
                            zoom_factor, 
                            is_v_flip,
                            is_h_flip,
                            is_transpose,
                            **argkws
                            ):
        
        bboxes *= zoom_factor
        
        
        roi_shape_z = [x*zoom_factor for x in roi_shape]
        if is_v_flip:
            bb = bboxes.copy()
            bboxes[..., 3] = roi_shape_z[0] - bb[..., 1]
            bboxes[..., 1] = roi_shape_z[0] - bb[..., 3]
            
        if is_h_flip:
            bb = bboxes.copy()
            bboxes[..., 2] = roi_shape_z[1] - bb[..., 0]
            bboxes[..., 0] = roi_shape_z[1] - bb[..., 2]
            
        if is_transpose:
            bboxes = bboxes[..., [1,0,3,2]]
            
        return bboxes
    
    @staticmethod
    def merge_bboxes(bboxes, cutoff_IoU = 0.5):
        if len(bboxes) <= 1:
            return bboxes
        
        for _ in range(3):
            box_areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
            n_boxes = len(bboxes)
            
            boxes2merge = []
            boxes2keep = np.ones(n_boxes, np.bool)
            for ibox1 in range(n_boxes - 1):
                box1 = bboxes[ibox1]
                box1_area = box_areas[ibox1]
                for ibox2 in range(ibox1+1, n_boxes):
                    box2 = bboxes[ibox2]
                    
                    #intersections
                    min_xy = np.maximum(box1[:2], box2[:2])
                    max_xy = np.minimum(box1[2:], box2[2:])
                    inter = (max_xy-min_xy).clip(min=0)
                    inter = inter[0] * inter[1]
                    
                    #union
                    box2_area = box_areas[ibox2]
                    union = box1_area + box2_area - inter
                    
                    IoU = inter/union
                    
                    if IoU > 0.5:
                        boxes2merge.append((ibox1, ibox2))
                        boxes2keep[ibox1] = False
                        boxes2keep[ibox2] = False
            if not boxes2merge:
                return bboxes
                
            mergedboxes = []
            for ibox1, ibox2 in boxes2merge:
                box1 = bboxes[ibox1]
                box2 = bboxes[ibox2]
                min_xy = np.minimum(box1[:2], box2[:2])
                max_xy = np.maximum(box1[2:], box2[2:])
                
                box_merged = np.concatenate((min_xy, max_xy))
                mergedboxes.append(box_merged)
            
            bboxes = np.concatenate((bboxes[boxes2keep], mergedboxes))
        else:
            return bboxes
        
    
    
    def _read_roi_rand(self):
        set_k = random.choice(self.data_types)
        _dat = random.choice(self.data[set_k])
        roi_mask, roi_full, widths, skels, cnts, cnts_bboxes, clusters_bboxes = _dat
        if self.is_clusters_bboxes:
            bboxes = clusters_bboxes
        else:
            bboxes = cnts_bboxes
        bboxes = np.array(bboxes).astype(np.float)
        
        
        assert not (roi_mask is None and roi_full is None)
        if (roi_full is not None) and (random.random() > 0.5):
            roi = roi_full.copy()
        else:
            roi = roi_mask.copy()
        skels, widths = skels.copy(), widths.copy()
        
        #randomly scale to match image or the whole range
        roi = (roi - self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
        params = self._transforms_params(roi, skels)
        
        roi = self._transforms_roi(roi, **params)
        
        skels, widths = self._transforms_skels(skels, widths, **params)
        
        if self.is_clusters_bboxes:
            bboxes = self._transforms_bbox(bboxes, **params)
        else:
            gap = np.ceil(np.max(widths/2, axis=1))[..., None]
            
            ss = skels - 0.5
            #get the boxes from the skeletons
            min_xy = ss.min(axis=1)
            min_xy = np.floor(min_xy) - gap
            min_xy = np.maximum(min_xy, 0)
            
            max_xy = ss.max(axis=1)
            max_xy = np.ceil(max_xy) + gap
            
            r_lims = np.array(roi.shape)[::-1][None]
            max_xy = np.minimum(r_lims, max_xy)
            
            bboxes = np.concatenate((min_xy, max_xy), axis=1)
            
        
        
        bboxes = self.merge_bboxes(bboxes)
        
        return roi, bboxes, skels
    
    
    def _build_rand_img(self):
        img = np.zeros((self.roi_size, self.roi_size), np.float32)
        img_skels = []
        img_bboxes = []
            
        n_rois = random.randint(*self.n_rois_lims)
        while n_rois > 0 or len(img_bboxes) == 0:
            roi, bboxes, skels = self._read_roi_rand()
            
            n_rois -= 1
            for _ in range(5):
                #try to position the current raw times
                ii = random.randint(0, self.roi_size - roi.shape[0])
                jj = random.randint(0, self.roi_size - roi.shape[1])
                roi_in_img = img[ii: ii + roi.shape[0], jj:jj + roi.shape[1]].view()
                
                if np.any(roi_in_img > 0):
                    continue
                
                
                
                skels[..., 1] += ii
                skels[..., 0] += jj
                
                bboxes[..., 1] += ii
                bboxes[..., 3] += ii
                bboxes[..., 0] += jj
                bboxes[..., 2] += jj
                
                
                
                #I am puting the skletons only in boxes that contain them completely
                
                roi_bboxes = []
                roi_skels = []
                for bbox in bboxes:
                    inside = (skels[...,0] >= bbox[0] ) & (skels[...,0] <= bbox[2])
                    inside &= (skels[...,1] >= bbox[1] ) & (skels[...,1] <= bbox[3])
                    
                    in_cluster = inside.mean(axis=1) >= 0.95
                    
                    skels_in = skels[in_cluster].copy()
                    
                    
                    if skels_in.size: 
                        
                        if self.fold_skeleton:
                            mid = skels_in.shape[1]//2
                            skels_in = np.concatenate((skels_in[:, :mid+1], skels_in[:, :mid-1:-1]), axis=0)
                        
                        roi_bboxes.append(bbox[None])
                        roi_skels.append(skels_in)
                    else:
                        
                        #if there is not at least one skeleton in everybox, there is a problem and I want to ignore this ROI...
                        break
                else:
                    roi_in_img += roi
                    img_skels += roi_skels
                    img_bboxes += roi_bboxes
                    break
                
                
                
                
        
        self._add_negative_examples(img)
        
        
        assert len(img_bboxes) == len(img_skels)
        
        return img, img_bboxes, img_skels
    
    
    def _add_negative_examples(self, img):
        if self.negative_data is None:
            return
        
        n_rois = random.randint(*self.n_rois_neg_lims)
        while n_rois > 0:
            roi = random.choice(self.negative_data)
            roi = roi.copy()
            
            roi = (roi - self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
            params = self._transforms_params(roi)
            roi = self._transforms_roi(roi, **params)
            
            n_rois -= 1
            for _ in range(5):
                #try to position the current raw times
                ii = random.randint(0, self.roi_size - roi.shape[0])
                jj = random.randint(0, self.roi_size - roi.shape[1])
                roi_in_img = img[ii: ii + roi.shape[0], jj:jj + roi.shape[1]].view()
                
                if np.any(roi_in_img > 0):
                    continue
                
                roi_in_img += roi
                break
    
    def _prepare_for_train(self, img, img_bboxes, img_skels):
        img = np.repeat(img[None].astype(np.float32), 3, axis=0)
        img = torch.from_numpy(img)
        
        if len(img_bboxes):
            bboxes = np.concatenate(img_bboxes, axis=0).astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), np.float32)
            
            
        labels = np.ones(len(bboxes))
            
        
        target = {}
        target['boxes'] = torch.from_numpy(bboxes)
        target['labels'] = torch.from_numpy(labels)
        target['keypoints'] = [torch.from_numpy(x.astype(np.float32)) for x in img_skels]
        
        return img, target
    
    
    def __getitem__(self, ind):
        img, img_bboxes, img_skels = self._build_rand_img()
        return self._prepare_for_train(img, img_bboxes, img_skels)
        
        
    
