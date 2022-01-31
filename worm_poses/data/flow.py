#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:11:22 2018

@author: avelinojaver
"""

from .prepare_data import read_data_files, read_negative_data
from .transforms import (AffineTransformBounded, Compose, RandomVerticalFlip, 
                        RandomHorizontalFlip, NormalizeIntensity, RandomIntensityExpansion,
                        RandomIntensityOffset, AddBlankPatch, PadToSize, ToTensor,
                        AddRandomLine, JpgCompression, RandomFlatField)
from .encode_PAF import get_part_affinity_maps

import numpy as np
import random
from pathlib import Path
import cv2
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
        if fold_skeleton:
            n_affinity_maps_out = (2*(n_segments//2 - PAF_seg_dist + 1) + 1)//2 #+ 1
        else:
            n_affinity_maps_out = n_segments - PAF_seg_dist
        
    else:
        n_affinity_maps_out = None
    
    
    return n_segments_out, n_affinity_maps_out


class SkelMapsFlow(Dataset):
    _valid_headtail = ['from_tierpsy']
    _head_ind = 2
    def __init__(self, 
                 root_dir, 
                 set2read = 'validation',
                 data_types = ['from_tierpsy', 'manual', 'from_NNv1'],
                 negative_src = ['from_tierpsy_negative.p.zip', 'from_hydra-bgnd_negative.p.zip'],
                 samples_per_epoch = 12288,
                 scale_int = (0, 255),
                 
                 roi_size = 362,
                 crop_size_lims = (25, 256),
                 negative_size_lims = (5, 180),
                 n_rois_lims = (1, 8),
                 int_expansion_range = (0.5, 1.3),
                 int_offset_range = None,
                 blank_patch_range = (1/8, 1/4),
                 mixup_probability = 0.,
                 
                 zoom_range = None,
                 
                 fold_skeleton = True,
                 
                 is_contour_PAF = True,
                 PAF_seg_dist = 5,
                 n_segments = 49,
                 
                 return_bboxes = False,
                 return_half_bboxes = False,
                 return_key_value_pairs = True
                 ):
        
        _dum = set(dir(self))
        self.return_key_value_pairs = return_key_value_pairs
        self.samples_per_epoch = samples_per_epoch
        
        self.roi_size = roi_size #if isinstance(roi_size, (tuple, list)) else (roi_size, roi_size)
        self.n_rois_lims = n_rois_lims
        self.scale_int = scale_int
        self.crop_size_lims = crop_size_lims
        self.negative_size_lims = negative_size_lims
        self.mixup_probability = mixup_probability
        
        self.return_bboxes = return_bboxes
        self.return_half_bboxes = return_half_bboxes
        
        self.fold_skeleton = fold_skeleton
        self.PAF_seg_dist = PAF_seg_dist
        self.is_contour_PAF = is_contour_PAF
        self.n_segments = n_segments
        self.n_segments_out, self.n_affinity_maps_out = get_outputs_sizes(n_segments, PAF_seg_dist, fold_skeleton)
        
        self.root_dir = Path(root_dir)
        self.set2read = set2read
        self.data_types = data_types
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        self.negative_src = [Path(x) for x in negative_src]
        self.negative_src = [x if x.exists() else root_dir / x for x in self.negative_src]
        
        
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
                        AddBlankPatch(blank_patch_range),
                        AddRandomLine()
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
        
        transforms = [RandomFlatField(self.roi_size),
                      JpgCompression(quality_range = (40, 90))]
        self.transforms_image = Compose(transforms)
        
        self.to_tensor = ToTensor()
        
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
    
    
    
    def _get_head_coords(self, skels, is_valid_ht):
        if isinstance(is_valid_ht, bool):
            is_valid_ht = np.array([is_valid_ht]*len(skels))
        head_coords = skels[:, self._head_ind].copy()
        head_coords[~is_valid_ht] = -1
        
        return head_coords, is_valid_ht
    
    def _get_random_worm(self, use_transforms = True):
        k, ii = random.choice(self.data_indexes)
        raw_data = self.data[k][ii]
        
        if raw_data['roi_full'] is None:
            image = raw_data['roi_mask']
        else:
            image = raw_data['roi_mask'] if random.random() >= 0.5 else raw_data['roi_full']
        target = {k:raw_data[k] for k in ('widths', 'skels')}
        
        
        #I am assuming 'from_tierpsy' has the correct head/tail orientation 
        is_valid_ht = k in self._valid_headtail
        if not is_valid_ht: 
            #If we don't know the correct orientation let's randomly switch so anyting else will be randomly switched
            if random.random() > 0.5:
                target['skels'] = target['skels'][:, ::-1] 
        
        target['is_valid_ht'] = is_valid_ht
        
        if use_transforms:
            image, target = self.transforms_worm(image, target)
        return image, target
    
    def _get_random_negative(self):
        negative_data = random.choice(self.data_negative)
        return self.transforms_negative(negative_data['image'], {})[0]
    
    def _get_random_mixup(self):
        #https://arxiv.org/abs/1710.09412
        roi_size = random.randint(*self.crop_size_lims)
        crop_transforms  = Compose(
                    [AffineTransformBounded(crop_size_lims = (roi_size, roi_size), rotation_range = (0, 0), interpolation = cv2.INTER_NEAREST),
                     RandomVerticalFlip(), 
                     RandomHorizontalFlip(),
                     PadToSize(roi_size)
                     ])
        
        dat = []
        while len(dat) < 2:
            image, target = self._get_random_worm(use_transforms = False)
            if target['skels'].shape[0] != 1:
                continue
            
            image, target = crop_transforms(image, target)
            
            image_r = image.astype(np.float32)
            
            good = image>0
            val_pix = image_r[good]
            bot, top = val_pix.min(), val_pix.max()
            image_r[good] = (val_pix - bot)/(top - bot + 1e-4)
            image_r[~good] = 1.
              
            dat.append((image_r, target))
        
        images, targets = zip(*dat)
        img_m = np.minimum(*images)
        intensity_factor = np.random.uniform(0.2, 0.95)
        intensity_offset = np.random.uniform(0, 1 - intensity_factor)
        img_m *= intensity_factor
        img_m += intensity_offset
        
        target_m = {}
        for k in targets[0]:
            dd = [x[k] for x in targets]
            dd = False if k == 'is_valid_ht' else np.concatenate(dd)
            target_m[k] = dd
        
        return img_m, target_m
    
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
        target = {'skels' : [], 'widths' : [], 'is_valid_ht' : []}
        
        n_rois = random.randint(*self.n_rois_lims)
        while n_rois > 0:

            if random.random() < self.mixup_probability:
                roi, roi_target = self._get_random_mixup()
            else:
                roi, roi_target = self._get_random_worm()
            
            
            
            n_rois -= 1
            
            corner = self._randomly_locate_roi(img, roi)
            if corner is not None:
                #Successful location!
                corner = np.array(corner)[None, None]
                target['skels'].append(corner + roi_target['skels'])
                target['widths'].append(roi_target['widths'])
                target['is_valid_ht'].append(roi_target['is_valid_ht'])
        
        target['is_valid_ht'] = [np.array([is_ht]*len(s)) for s, is_ht in zip(target['skels'], target['is_valid_ht'])]
        target = {k : np.concatenate(v) for k, v in target.items()}
        assert target['skels'].shape[0] == target['is_valid_ht'].shape[0]
        
        
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
        is_valid_ht = target['is_valid_ht']
        
        widths[widths<1] = 1 #clip to have at lest one of width
        
        if skels.shape[1] != self.n_segments:
            inds = np.linspace(0, skels.shape[1] -1, self.n_segments).round().astype(np.int)
            skels = skels[:, inds]
            widths = widths[:, inds]
        
        skels = skels.round()
        skels = np.clip(skels, 0, self.roi_size-1) #ensure the skeletons are always in the image limits
        heads, is_valid_ht = self._get_head_coords(skels, is_valid_ht)
        if self.fold_skeleton:
            heads = np.concatenate((heads, -np.ones_like(heads)))
        
        
        if self.return_bboxes:
            target_out = self.skels2bboxes(skels, widths)
        elif self.return_half_bboxes:
            target_out = self.skels2halfbboxes(skels, widths)
        else:
            target_out = self.skels2PAFs(skels, widths)
            heads = heads[heads[:, 0]>=0]
        
        target_out['heads'] = heads
        
        if not self.fold_skeleton:
            target_out['is_valid_ht'] = is_valid_ht
        
        image_out = np.clip(image, 0, 1)
        image_out, target_out = self.transforms_image(image_out, target_out)
        
        image_out, target_out = self.to_tensor(image_out[None], target_out)
        
        if self.return_key_value_pairs:
            out = [x for d in target_out.items() for x in d]
            return (image_out, *out)
        else:
            return image_out, target_out
   
    
    def skels2PAFs(self, skels, widths):
        skels = skels.astype(np.int)
        
        if not self.is_contour_PAF:
            mid = skels.shape[1]//2 + 1
            widths = [max(2, w[mid]/2) for w in widths]
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
    
    @staticmethod
    def _skels2bboxes(skels, widths, roi_size, fold_skeleton):
        skels = skels.astype(np.float32)
        
        gap = np.ceil(np.max(widths/2, axis=1))[..., None]
        
        bbox_l = skels.min(axis = 1) - gap
        bbox_l[bbox_l<0] = 0
        
        bbox_r = skels.max(axis = 1) + gap
        s = roi_size-1
        bbox_r[bbox_r>s] = s
        
        bboxes = np.concatenate((bbox_l, bbox_r), axis = 1)
        bboxes = _merge_bboxes(bboxes)
        
        keypoints = []
        
        
        for bbox in bboxes:
            inside = (skels[...,0] >= bbox[0] ) & (skels[...,0] <= bbox[2])
            inside &= (skels[...,1] >= bbox[1] ) & (skels[...,1] <= bbox[3])
        
            in_cluster = inside.mean(axis=1) >= 0.95
            skels_in = skels[in_cluster]
            
            if fold_skeleton:
                mid = skels_in.shape[1]//2
                skels_in = np.concatenate((skels_in[:, :mid+1], skels_in[:, :mid-1:-1]))
            keypoints.append(skels_in)
        labels = np.ones(len(bboxes), np.float32)
        
        target = dict(
            boxes = bboxes,
            labels = labels,
            keypoints = keypoints
            )
        return target
    
    def skels2bboxes(self, skels, widths):
        return self._skels2bboxes(skels, widths, self.roi_size, self.fold_skeleton)
        
    
    def skels2halfbboxes(self, skels, widths):
        mid = self.n_segments//2
        
        skels = np.concatenate((skels[:, :mid+1], skels[:, :mid-1:-1]))
        widths = np.concatenate((widths[:, :mid+1], widths[:, :mid-1:-1]))
        
        return self._skels2bboxes(skels, widths, self.roi_size, False)
        
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
        
        target['is_valid_ht'] = k in self._valid_headtail
        
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
    
    
    #root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
    #root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training_filtered/'
    #%%
    #argkws = dict(PAF_seg_dist = 1, n_segments = 15)
    argkws = dict(PAF_seg_dist = 1, n_segments = 15, fold_skeleton = True, is_contour_PAF = False, n_rois_lims = (1, 8), mixup_probability = 0.25)
    
    gen = SkelMapsFlow(root_dir = root_dir, return_key_value_pairs = False,  **argkws)
    # gen_val = SkelMapsFlowValidation(root_dir = root_dir, return_key_value_pairs = False, **argkws)
    
    # gen_boxes = SkelMapsFlow(root_dir = root_dir, return_key_value_pairs = False, return_bboxes = True)
    # gen_half_boxes = SkelMapsFlow(root_dir = root_dir, return_key_value_pairs = False, return_half_bboxes = True)
    #%%\
    
    
    img_m, target_m = gen._get_random_mixup()
    
    
    #fig, axs = plt.subplots(1,3, sharex  = True, sharey = True)
    #axs[0].imshow(images[0], cmap = 'gray', vmin=0., vmax = 1.)
    #axs[1].imshow(images[1], cmap = 'gray', vmin=0., vmax = 1.)
    
    plt.figure()
    plt.imshow(img_m, cmap = 'gray', vmin=0., vmax = 1.)
    for skel in target_m['skels']:
        plt.plot(skel[:, 0], skel[:, 1])
    
    
    
        
    #%%
    for ind in range(5):
        image, target = gen._get_random_worm()
        
        
        fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
        
        
        print(image.min(), image.max())
        for ax in axs:
            ax.imshow(image, cmap = 'gray')
            ax.axis('off')
        
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1])
        
        
    #%%
    # for _ in range(5):
    #     roi = gen._get_random_negative()
    #     plt.figure()
    #     plt.imshow(roi, cmap = 'gray')
    #%%
    for ind in tqdm.trange(5):
        image, target = gen._build_rand_img()
        image, target = gen._prepare_output(image, target)
        
        img = image[0]
        
        pafs_abs = np.linalg.norm(target['PAF'], axis = 1)
        pafs_head = pafs_abs[0]
        pafs_tail = pafs_abs[-1]
        pafs_max = pafs_abs.max(axis=0)
        
        fig, axs = plt.subplots(1,5, figsize = (15, 5), sharex = True, sharey = True)
        for ax in axs:
            ax.axis('off')
        
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        axs[2].imshow(pafs_head)
        axs[3].imshow(pafs_tail)
        
        axs[4].imshow(pafs_max)
        
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1], '.-')
        
        p = target['heads']
        p = p[p[:, 0]>=0] 
        axs[1].plot(p[:, 0], p[:, 1], 'or')
    a
    
    skels = target['skels'].detach().numpy()
    
    PAF_switched = -torch.flip(target['PAF'], dims = (0,))
    PAF = target['PAF']
    
    
    for ii, paf in enumerate(PAF):
        fig, axs = plt.subplots(1, 2, figsize = (20, 10))
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(np.linalg.norm(paf, axis = 0))
        
        
        #if gen.fold_skeleton:
        if (ii < gen.n_affinity_maps_out) or not gen.fold_skeleton:
            s1 = skels[:, ii]
            s2 = skels[:, ii + gen.PAF_seg_dist]
        else:
            mid = len(skels)//2
            ind = max(2, gen.PAF_seg_dist //2 + 1)
            s1 = skels[:mid, -ind]
            s2 = skels[mid:, -ind]
                
                
        sx = (s1[:, 0], s2[:, 0])
        sy = (s1[:, 1], s2[:, 1])
        
        for ax in axs:
            ax.plot(sx, sy, 'r.-')
            ax.axis('off')
    
        #%%
                
    for ind in tqdm.trange(1):
        image, target = gen_boxes._build_rand_img()
        image, target = gen_boxes._prepare_output(image, target)
        
        img = image[0]
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        ax.imshow(img, cmap = 'gray')
        
        for bbox, ss, p in zip(target['boxes'], target['keypoints'], target['heads']):
                
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            
            for s in ss:
                plt.plot(s[:, 0], s[:, 1])
            
            if p[0] >= 0:
                plt.plot(p[0], p[1], 'or')
    
    #%%        
    for ind in tqdm.trange(5):
        image, target = gen_half_boxes._build_rand_img()
        image, target = gen_half_boxes._prepare_output(image, target)
        
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
    for ind in range(10, 20):
        image, target =  gen_val._read_worm(ind)
        image, target = gen_val._prepare_output(image, target)
        img = image[0] 
        
        fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
     
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        for skel in target['skels']:
            axs[1].plot(skel[:, 0], skel[:, 1], '.-')
            
        if target['heads'].shape[0] >= 0:
            h = target['heads']
            axs[1].plot(h[:, 0], h[:, 1], 'or')
