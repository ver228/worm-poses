#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:11:22 2018

@author: avelinojaver
"""

from .prepare_data import read_data_files, read_negative_data
from .transforms import (AffineTransformBounded, Compose, RandomVerticalFlip, 
                        RandomHorizontalFlip, NormalizeIntensity, RandomIntensityExpansion,
                        RandomIntensityOffset, AddBlankPatch, PadToSize, ToTensor)
from .encode_PAF import get_part_affinity_maps

import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

def collate_simple(batch):
    return tuple(map(list, zip(*batch)))

def collate_to_dict(batch):
    #this is a way to speed up the data transfer using the pytorch multiprocessing
    #passing dictionaries or list apprently create an copy of the object instead of passing by reference
    #using a numpy array instead solve the process. Therefore for the targets i will use 
    #a `key` `value` pairs similar to the MATLAB arguemnts (image, key1, value1, key2, value2, ...)
    images = []
    targets = []
    
    for dat in batch:
        images.append(dat[0])
        target = dict(list(zip(dat[1::2], dat[2::2])))
        targets.append(target)
    
    return torch.stack((images)), tuple(targets)    


def _merge_bboxes(bboxes, cutoff_IoU = 0.5):
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


def get_outputs_sizes(n_segments, PAF_seg_dist, fold_skeleton):
    
    n_segments_out = n_segments // 2 + 1 if fold_skeleton else n_segments
    
    if PAF_seg_dist:
        n_affinity_maps = 2*(n_segments//2 - PAF_seg_dist + 1) + 1
        n_affinity_maps_out = n_affinity_maps // 2 + 1 if fold_skeleton else n_affinity_maps
    else:
        n_affinity_maps_out = None
    
    
    return n_segments_out, n_affinity_maps_out


class SkelMapsFlow(Dataset):
    _valid_headtail = ['from_tierpsy']
    
    def __init__(self, 
                 root_dir, 
                 set2read = 'validation',
                 data_types = ['from_tierpsy', 'manual'],
                 negative_src = 'from_tierpsy_negative.p.zip',
                 samples_per_epoch = 12288,
                 scale_int = (0, 255),
                 
                 roi_size = 256,
                 crop_size_lims = (50, 180),
                 negative_size_lims = (5, 180),
                 n_rois_lims = (1, 4),
                 int_expansion_range = (0.7, 1.3),
                 int_offset_range = (-0.2, 0.2),
                 blank_patch_range = (1/8, 1/3),
                 zoom_range = None,
                 
                 fold_skeleton = True,
                 
                 PAF_seg_dist = 5,
                 n_segments = 49,
                 
                 return_bboxes = False,
                 return_key_value_pairs = True
                 ):
        
        _dum = set(dir(self))
        self.return_key_value_pairs = return_key_value_pairs
        self.samples_per_epoch = samples_per_epoch
        
        self.roi_size = roi_size #if isinstance(roi_size, (tuple, list)) else (roi_size, roi_size)
        self.n_rois_lims = n_rois_lims
        self.scale_int = scale_int
        
        self.return_bboxes = return_bboxes
        
        self.fold_skeleton = fold_skeleton
        self.PAF_seg_dist = PAF_seg_dist
        self.n_segments = n_segments
        
        self.n_segments_out, self.n_affinity_maps_out = get_outputs_sizes(n_segments, PAF_seg_dist, fold_skeleton)
        
        self.root_dir = Path(root_dir)
        self.set2read = set2read
        self.data_types = data_types
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        negative_src = Path(negative_src)
        self.negative_src = negative_src if negative_src.exists() else root_dir / negative_src
        
        
        self.data = read_data_files(root_dir, set2read, data_types)
        self.data_negative = read_negative_data(self.negative_src)    
        
        self.data_types = list(self.data.keys())
        self.data_indexes = [(k, ii) for k,val in self.data.items() for ii in range(len(val))]
        
        
        
        transforms = [AffineTransformBounded(crop_size_lims = crop_size_lims, zoom_range = zoom_range),
                        RandomVerticalFlip(), 
                        RandomHorizontalFlip(),
                        NormalizeIntensity(scale_int),
                        RandomIntensityExpansion(int_expansion_range), 
                        RandomIntensityOffset(int_offset_range),
                        AddBlankPatch(blank_patch_range)
                        ]
        self.transforms_worm = Compose(transforms)
        
        transforms = [AffineTransformBounded(crop_size_lims = negative_size_lims, zoom_range = zoom_range),
                        RandomVerticalFlip(), 
                        RandomHorizontalFlip(),
                        NormalizeIntensity(scale_int),
                        RandomIntensityExpansion(int_expansion_range), 
                        RandomIntensityOffset(int_offset_range)
                        ]
        self.transforms_negative = Compose(transforms)
        
        self.to_tensor = ToTensor()
        
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
    
    def _get_random_worm(self):
        k, ii = random.choice(self.data_indexes)
        raw_data = self.data[k][ii]
        
        if raw_data['roi_full'] is None:
            image = raw_data['roi_mask']
        else:
            image = raw_data['roi_mask'] if random.random() >= 0.5 else raw_data['roi_full']
        target = {k:raw_data[k] for k in ('widths', 'skels')}
        
        #I am assuming 'from_tierpsy' has the correct head/tail orientation so anyting else will be randomly switched
        if not k in self._valid_headtail: 
            if random.random() > 0.5:
                target['skels'] = target['skels'][:, ::-1] 
        
        image, target = self.transforms_worm(image, target)
        
        return image, target
    
    def _get_random_negative(self):
        negative_data = random.choice(self.data_negative)
        return self.transforms_negative(negative_data['image'], {})[0]
    
    @staticmethod
    def _randomly_locate_roi(image, roi, n_trials = 5):
        assert image.shape[0] > roi.shape[0] and image.shape[1] > roi.shape[1]
        
        for _ in range(5):
            #try to position the current raw times
            ii = random.randint(0, image.shape[0] - roi.shape[0])
            jj = random.randint(0, image.shape[1] - roi.shape[1])
            roi_in_img = image[ii: ii + roi.shape[0], jj:jj + roi.shape[1]].view()
            
            if np.any(roi_in_img > 0):
                continue
            
            roi_in_img += roi
            
            return (jj, ii)
            
        else:
            return None
    
    def _build_rand_img(self):
        img = np.zeros((self.roi_size, self.roi_size), np.float32)
        target = {'skels' : [], 'widths' : []}
        
        n_rois = random.randint(*self.n_rois_lims)
        while n_rois > 0:

            roi, roi_target = self._get_random_worm()
            n_rois -= 1
            
            corner = self._randomly_locate_roi(img, roi)
            if corner is not None:
                #Successful location!
                corner = np.array(corner)[None, None]
                target['skels'].append(corner + roi_target['skels'])
                target['widths'].append(roi_target['widths'])
        target = {k : np.concatenate(v) for k, v in target.items()}
        
        
        if self.data_negative is not None:
            n_rois = random.randint(*self.n_rois_lims)
            while n_rois > 0:
                roi = self._get_random_negative()
                n_rois -= 1
                corner = self._randomly_locate_roi(img, roi)
        
        return img, target
        
    def _prepare_output(self, image, target):
        
        skels = target['skels']
        widths = target['widths']
        widths[widths<1] = 1 #clip to have at lest one of width
        
        if skels.shape[1] != self.n_segments:
            inds = np.linspace(0, skels.shape[1] -1, self.n_segments).round().astype(np.int)
            skels = skels[:, inds]
            widths = widths[:, inds]
        
        skels = skels.round()
        skels = np.clip(skels, 0, self.roi_size-1) #ensure the skeletons are always in the image limits
        
        if self.return_bboxes:
            target_out = self.skels2bboxes(skels, widths)
        else:
            target_out = self.skels2PAFs(skels, widths)
        
        image_out = np.clip(image, 0, 1)[None]
        image_out, target_out = self.to_tensor(image_out, target_out)
        
        if self.return_key_value_pairs:
            out = [x for d in target_out.items() for x in d]
            return (image_out, *out)
        else:
            return image_out, target_out
   
    
    def skels2PAFs(self, skels, widths):
        skels = skels.astype(np.int)
        PAF = get_part_affinity_maps(skels, 
                                          widths, 
                                          (self.roi_size, self.roi_size), 
                                          self.PAF_seg_dist,
                                          fold_skeleton = self.fold_skeleton
                                          )
        assert PAF.shape == (self.n_affinity_maps_out, 2, self.roi_size, self.roi_size)
        PAF = PAF.astype(np.float32)
        
        if self.fold_skeleton:
            mid = self.n_segments//2
            skels = np.concatenate((skels[:, :mid+1], skels[:, :mid-1:-1]))
        assert skels.shape[1] == self.n_segments_out
        
        target = dict(skels = skels, PAF = PAF)
        return target
    
    
    def skels2bboxes(self, skels, widths):
        skels = skels.astype(np.float32)
        
        gap = np.ceil(np.max(widths/2, axis=1))[..., None]
        
        bbox_l = skels.min(axis = 1) - gap
        bbox_l[bbox_l<0] = 0
        
        bbox_r = skels.max(axis = 1) + gap
        s = self.roi_size-1
        bbox_r[bbox_r>s] = s
        
        bboxes = np.concatenate((bbox_l, bbox_r), axis = 1)
        bboxes = _merge_bboxes(bboxes)
        
        keypoints = []
        
        mid = self.n_segments//2
        for bbox in bboxes:
            inside = (skels[...,0] >= bbox[0] ) & (skels[...,0] <= bbox[2])
            inside &= (skels[...,1] >= bbox[1] ) & (skels[...,1] <= bbox[3])
        
            in_cluster = inside.mean(axis=1) >= 0.95
            skels_in = skels[in_cluster]
            
            if self.fold_skeleton:
                skels_in = np.concatenate((skels_in[:, :mid+1], skels_in[:, :mid-1:-1]))
            keypoints.append(skels_in)
        
        labels = np.ones(len(bboxes), np.float32)
        
        target = dict(
            boxes = bboxes,
            labels = labels,
            keypoints = keypoints
            )
        
        
        return target
    
    def __len__(self):
        return self.samples_per_epoch


    def __getitem__(self, ind):
        image, target = self._build_rand_img()
        return self._prepare_output(image, target)
    
    
    
class SkelMapsFlowValidation(SkelMapsFlow):
    def __init__(self, *args, **argkws):
        super().__init__(*args, **argkws)
        
        transforms = [NormalizeIntensity(self.scale_int),
                      PadToSize(self.roi_size)]
        self.transforms_full = Compose(transforms)
    
    def _read_worm(self, ind):
        
        k, ii = self.data_indexes[ind]
        raw_data = self.data[k][ii]
        image = raw_data['roi_mask']
        target = {k:raw_data[k] for k in ('widths', 'skels')}
        
        image, target = self.transforms_full(image, target)
        return image, target
    
    def __getitem__(self, ind):
        image, target = self._read_worm(ind)
        return self._prepare_output(image, target)
        
    
    def __len__(self):
        return len(self.data_indexes)
    
    
if __name__ == '__main__':
    import tqdm
    import matplotlib.pylab as plt
    from matplotlib import patches
    
    root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    
    argkws = dict(PAF_seg_dist = 2, n_segments = 17)
    
    gen = SkelMapsFlow(root_dir = root_dir, return_key_value_pairs = False, **argkws)
    gen_val = SkelMapsFlowValidation(root_dir = root_dir, return_key_value_pairs = False, **argkws)
    
    gen_boxes = SkelMapsFlow(root_dir = root_dir, return_key_value_pairs = False, return_bboxes = True)
    
    #%%
    for ind in range(5):
        image, target = gen._get_random_worm()
        fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
        
        axs[0].imshow(image, cmap = 'gray')
        axs[1].imshow(image, cmap = 'gray')
        
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1])
        
        for ax in axs:
            ax.axis('off')
    
    
    
    #%%
    for _ in range(5):
        roi = gen._get_random_negative()
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
    #%%
    for ind in tqdm.trange(5):
        image, target = gen._build_rand_img()
        image, target = gen._prepare_output(image, target)
        
        img = image[0]
        
        pafs_abs = np.linalg.norm(target['PAF'], axis = 1)
        pafs_head = pafs_abs[0]
        pafs_tail = pafs_abs[-1]
        pafs_max = pafs_abs.max(axis=0)
        
        fig, axs = plt.subplots(1,5, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        axs[2].imshow(pafs_head)
        axs[3].imshow(pafs_tail)
        
        axs[4].imshow(pafs_max)
        
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1], '.-')
            
        #%%
                
    for ind in tqdm.trange(5):
        image, target = gen_boxes._build_rand_img()
        image, target = gen_boxes._prepare_output(image, target)
        
        img = image[0]
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap = 'gray')
        
        
        for bbox, ss in zip(target['boxes'], target['keypoints']):
                
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            
            for s in ss:
                plt.plot(s[:, 0], s[:, 1])

    #%%
        
      #%%
    for ind in range(480, 485):#[491, 495]:
        image, target =  gen_val._read_worm(ind)
        image, target = gen_val._prepare_output(image, target)
        img = image[0] 
        
        fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
     
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1], '.-')
            
            