#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:54:56 2019

@author: avelinojaver
"""

#import tqdm
#import matplotlib.pylab as plt
#plt.plot([1,5,6])

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from worm_poses.models import CPM_PAF, PretrainedBackBone, get_preeval_func
from worm_poses.encoders import maps2skels

import tqdm
import tables
import pandas as pd
import torch
import math
import numpy as np
import cv2
import multiprocessing as mp
#%%

def load_model(model_path):
    
    n_segments = 25
    n_affinity_maps = 21
    bn = model_path.parent.name
    
    preeval_func = get_preeval_func(bn)
    
    if 'vgg19' in bn:
        backbone = PretrainedBackBone('vgg19')
    elif 'resnet50' in bn:
        backbone = PretrainedBackBone('resnet50')
    else:
        backbone = None
    
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True,
                             backbone = backbone
                             )
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, preeval_func

def get_device(cuda_id):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    return device


class IterateROIs():
    def __init__(self, 
                 mask_file, 
                 rows2read,
                 roi_size = 96,
                 batch_size = 16,
                 ):
        
        self.mask_file = mask_file
        self.rows2read = rows2read
        self.roi_size = roi_size
        self.batch_size = batch_size
        
        self.batch = []
    
    def __iter__(self):
        with tables.File(mask_file, 'r') as fid:
            masks = fid.get_node('/mask')
            rows2check_g = rows2check.groupby('frame_number')
            for frame_number, frame_data in rows2check_g:
                img = masks[frame_number]
                
                for irow, row in frame_data.iterrows():
                    roi_size = int(row['roi_size'])
                    ll = int(math.ceil(roi_size/2))
                    xl = int(max(row['coord_x'] - ll, 0))
                    yl = int(max(row['coord_y'] - ll, 0))
                    roi = img[yl:yl + roi_size, xl:xl + roi_size]
                    roi, zoom = self._prepare_roi(roi)
                    
                
                    corner = (xl, yl)
                    self.batch.append((frame_number, corner, zoom, roi))
                    
                    if len(self.batch) >= self.batch_size:
                        yield self.emit_batch()
            if self.batch:
                yield self.emit_batch()
    
    def emit_batch(self):
        out = map(np.array, zip(*self.batch))
        self.batch = []
        return out
    
    def _prepare_roi(self, roi):
        l_max = max(roi.shape)
        
        _zoom = 1.
        if l_max > self.roi_size:
            _zoom = self.roi_size/l_max
            roi = cv2.resize(roi, dsize = (0,0), fx = _zoom, fy = _zoom)
            assert max(roi.shape) <= self.roi_size
            
        dd = [self.roi_size - x for x in roi.shape]
        _pad = [(math.floor(x/2), math.ceil(x/2)) for x in dd]
        roi = np.pad(roi, _pad, 'constant', constant_values = 0)
        return roi, _zoom
        
    def __len__(self):
        return int(math.ceil(len(self.rows2read)/self.batch_size))



def iterate_rois_queue(mask_file,  rows2check, roi_size, batch_size, queue):
    gen = IterateROIs(mask_file, rows2check, roi_size, batch_size)
    for batch in tqdm.tqdm(gen):
        queue.put(batch)
    queue.put(None)
    
#%%
if __name__ == '__main__':
    
    bn = 'allPAF_PAF+CPM_maxlikelihood_20190620_003150_adam_lr0.0001_wd0.0_batch48'
    
#    model_path = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/models' / bn / 'checkpoint-199.pth.tar'#'checkpoint.pth.tar'
#    mask_file = '/Users/avelinojaver/Imperial College London/Feriani, Luigi - Mating_for_Avelino/Mating_Assay/Mating_Assay_220618/MaskedVideos/Set2_N2_PS3398/Set2_N2_PS3398_Ch1_22062018_122752.hdf5'
#    
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/allPAF' / bn / 'checkpoint-199.pth.tar'#'checkpoint.pth.tar'
    mask_file = Path.home() / 'workspace/WormData/screenings/mating_videos/Mating_Assay/Mating_Assay_220618/MaskedVideos/Set2_N2_PS3398/Set2_N2_PS3398_Ch1_22062018_122752.hdf5'
    
    cuda_id = 0
    roi_size = 128
    batch_size = 64
    
    mask_file = Path(mask_file)
    bn = mask_file.stem
    skels_file = mask_file.parent / (bn + '_skeletons.hdf5')
    if not skels_file.exists():
        dname = str(mask_file.parent).replace('/MaskedVideos/', '/Results/') 
        skels_file = Path(dname) / (bn + '_skeletons.hdf5')
        assert skels_file.exists()
    
    with pd.HDFStore(skels_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']    
    
     
    is_unskel = ~trajectories_data['is_good_skel'] 
    if 'skel_outliers_flag'in trajectories_data:
        is_unskel &= trajectories_data['skel_outliers_flag'] > 0
    rows2check = trajectories_data[is_unskel]
    
    #%%
    device = get_device(cuda_id)
    model, preeval_func = load_model(model_path)
    model = model.to(device)
    
    batch_queue = mp.Queue(2)
    d = mp.Process(target = iterate_rois_queue, args= (mask_file,  rows2check, roi_size, batch_size, batch_queue))
    d.daemon = True
    d.start()
    
    
    predicted_skeletons = []
    while True:
        out = batch_queue.get()
        if out is None:
            break
        frame_numbers, corners, zooms, rois = out
        
        X = torch.from_numpy(rois).float()
        X /= 255
        X = X.unsqueeze(1)
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            
            cpm_maps_r, paf_maps_r = outs[-1]
            cpm_maps_r = preeval_func(cpm_maps_r)
        
            cpm_maps_r = cpm_maps_r.detach().cpu().numpy()
            paf_maps_r = paf_maps_r.detach().cpu().numpy()
        
        
        for t, corner, z, cpm_map, paf_maps in zip(frame_numbers, corners, zooms, cpm_maps_r, paf_maps_r):
            skels =  maps2skels(cpm_map, paf_maps)
            for skel in skels:
                skel = skel/z + corner[None]
                predicted_skeletons.append((t, skel))
        
        
                
#            with tables.File(mask_file, 'r') as fid:
#                img = fid.get_node('/mask')[t]
#            import matplotlib.pylab as plt
#            plt.figure()
#            plt.imshow(img, cmap = 'gray')
#            for skel in skels:
#                skel = skel/z + corner[None]
#                
#                plt.plot(skel[..., 0], skel[..., 1], 'r')
#                #%%
#            break
#        break
