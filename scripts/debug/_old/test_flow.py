#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:51:15 2019

@author: avelinojaver
"""

import sys
import numpy as np

from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.flow import *
from worm_poses.models.detection_torch import keypointrcnn_resnet50_fpn

import tqdm    
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from matplotlib import patches



#%%
if __name__ == '__main__':

    
    roi_size = 96
    file_list_csv = None
    
#    root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190627_113423/'
#    data_types = ['manual', 'from_tierpsy']
#    set2read = 'train'
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    data_types = ['from_tierpsy', 'manual']#['manual']#
    set2read = 'validation'
    
    
#%%
    gen = RandomFlowDetection(root_dir = root_dir,
                              data_types = data_types,
                              set2read = set2read,
                             roi_size = 512, 
                             skel_size_lims = (90, 150),
                             n_rois_lims = (1, 5),
                             n_rois_neg_lims = (1, 5),
                             fold_skeleton = False,
                             negative_file = 'negative_from_tierpsy.p.zip',
                             is_clusters_bboxes = False
                            )
#    gen = SkelMapsSimpleFlow(root_dir = root_dir,
#                              data_types = data_types,
#                              set2read = set2read,
#                             roi_size = 512
#                            )
    
    #%%
    
#    gen.return_skeletons = False
#    val_loader = DataLoader(gen, 
#                            batch_size = 4, 
#                            num_workers = 1,
#                            collate_fn = collate_simple
#                            )
#    
#    for X, targets in tqdm.tqdm(val_loader, desc = 'Data Loader'):
#        break
#    
#    model = keypointrcnn_resnet50_fpn(num_classes = 2, 
#                                      num_keypoints = 25,
#                                    min_size = 512,
#                                    max_size = 512,
#                                    image_mean = [0, 0, 0],
#                                    image_std = [1., 1., 1.],
#                                    pretrained = False
#                                    )
#    model.train()
#    losses = model(X, targets = targets)
#    loss = sum([x for x in losses.values()])
#    loss.backward()
    #%%
    for _ in tqdm.tqdm(range(20)):
        
        img, target = gen[0]
        img = img.detach().numpy()
        img = np.rollaxis(img, 0 ,3 )
        
        fig, ax = plt.subplots(1,1)
        plt.imshow(img, cmap='gray')
        
        for lab, bbox, skels in zip(target['labels'], target['boxes'], target['keypoints']):
            bbox = bbox.detach().numpy()
            
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            s = str(len(skels)/2)
            ax.text(x, y, s, color='cyan', fontsize=12)
        
       
            for skel in skels:
                skel = skel.detach().numpy()
                ax.plot(skel[:,0], skel[:,1], 'r')
                ax.plot(skel[0,0], skel[0,1], 'xr')
                
    
    #%%
#    for _ in tqdm.tqdm(gen):
#        pass
    
            #%%
            
        
    
#    #this is a fast method to find what boxes have some intersection  
#    min_xy = np.maximum(bboxes[:, None, :2], bboxes[None, :, :2])
#    max_xy = np.minimum(bboxes[:, None, 2:], bboxes[None, :, 2:])
#    
#    #negative value here means no inteserction
#    inter = (max_xy-min_xy).clip(min=0)
#    inter = inter[...,0] * inter[...,1]
#    
#    
#    areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
#    
#    
#    
#    union = areas[:,None] + areas[None] - inter
#    
#    jaccard = inter/union
#    
    #return jaccard


    #%%
#    for n_figs, (X,Y) in enumerate(gen):
#        #%%
#        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
#        axs[0].imshow(X[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
#        
#        mm = np.max(Y[0], axis=0)
#        axs[1].imshow(mm)
#        
#        #%%
#        if n_figs > 5:
#            break
    
#%%
    
    gen = SkelMapsSimpleFlow(root_dir = root_dir,
                             set2read = set2read,
                             roi_size = roi_size, 
                             return_raw_skels = True,
                             is_fixed_width = True,
                             min_sigma = 1.,
                             fold_skeleton = False
                             )
    
    for n_figs, ind in enumerate(tqdm.tqdm(range(500, len(gen), 1))):#enumerate(range(500, len(gen), 10)):
        X,Y = gen[ind]
        assert X.shape == (1, gen.roi_size, gen.roi_size)
        assert not (np.isnan(Y[0]).any() or np.isnan(Y[1]).any())
        
        
        fig, axs = plt.subplots(1, 3, sharex = True, sharey = True)
        axs[0].imshow(X[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
        
        mm = np.max(Y[0], axis=0)
        axs[1].imshow(mm)
        
        mm = np.max(Y[0], axis=0)
        axs[2].imshow(Y[0][0])
        
        skels_true = Y[-1]
        for ss in skels_true:
            axs[0].plot(ss[:, 0], ss[:, 1], '.-')
            axs[0].plot(ss[25, 0], ss[25, 1], 'o')
        
        if n_figs > 5:
            break
        
        #%%
    
    
    
#    val_loader = DataLoader(gen, 
#                            batch_size = 4, 
#                            num_workers = 1,
#                            collate_fn = collate_with_skels
#                            )
#    
#    for X, Y in tqdm.tqdm(val_loader, desc = 'Test collate with true skeletons'):
#        assert Y[2][0].shape[1] == 49
        
    #%%
    
    flow_args = dict(
            
            scale_int = (0, 255),
             
            n_segments = 49,
            skel_size_lims = (60, 200),
         
            n_rois_lims = (1, 1),
             
            int_aug_offset = (-0.2, 0.2),
            int_aug_expansion = (0.7, 1.5),
         
            width2sigma = -1,#0.25,
            min_sigma = 1.
            )
    
    
    gen = SkelMapsRandomFlow(root_dir = root_dir,
                              set2read = set2read,
                              roi_size = roi_size, 
                              
                              **flow_args)

    for ii in tqdm.tqdm(range(10)):
        roi, _out = gen[ii]
        
        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
        axs[0].imshow(roi[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
        
        mm = np.max(_out[0], axis=0)
        axs[1].imshow(mm)