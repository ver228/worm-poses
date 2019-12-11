#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
#%%
import torch
from worm_poses.trainer import get_device
from worm_poses.flow import RandomFlowDetection
from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
#%%

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    from matplotlib import patches
    
    
    #bn = 'detection-singles_fasterrcnn_20190628_174616_adam_lr0.0001_wd0.0_batch16'
    #bn = 'detection-clusters_keypointrcnn_20190628_174659_adam_lr0.0001_wd0.0_batch8'
    #bn = 'detection-singles_keypointrcnn_20190701_222513_adam_lr0.0001_wd0.0_batch8'
    
    #bn = 'detection-singles_keypointrcnn_20190702_132349_adam_lr0.0001_wd0.0_batch8'
    bn = 'detection-singles_keypointrcnn_20190703_085950_adam_lr0.0001_wd0.0_batch8'
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'checkpoint.pth.tar'
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    roi_size = 512
    if 'keypointrcnn' in bn:
        model = keypointrcnn_resnet50_fpn(
                                        num_classes = 2, 
                                        num_keypoints = 25,
                                        min_size = roi_size,
                                        max_size = roi_size,
                                        image_mean = [0, 0, 0],
                                        image_std = [1., 1., 1.]
                                        )
    elif 'fasterrcnn' in bn:
        model = fasterrcnn_resnet50_fpn(
                                    num_classes = 2, 
                                    min_size = roi_size,
                                    max_size = roi_size,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.]
                                    )
        
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    roi_size = 512
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    flow_args = dict(
                     is_clusters_bboxes = True,
                     skel_size_lims = (90, 150),
                     n_rois_lims = (1, 5),
                     n_rois_neg_lims = (1, 5),
                     negative_file = 'negative_from_tierpsy.p.zip',
                     int_aug_offset = (-0.2, 0.2),
                     int_aug_expansion = (0.7, 1.5)
                     
                     )
    
    
    gen = RandomFlowDetection(root_dir = root_dir, 
                                        set2read = 'validation',
                                        roi_size = roi_size,
                                        epoch_size = 100,
                                        **flow_args
                                        )
    
    #%%
    for _ in range(5):
        image, target = gen[0]
        
        
        X = image[None].float()
        predictions = model(X)
        
        
        img = image.detach().numpy()
        img = np.rollaxis(img, 0, 3)
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap='gray')
        
        
        for bbox in predictions[0]['boxes']:
            bbox = bbox.detach().numpy()
            
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        
        
        if 'keypoints' in predictions[0]:
            maps = predictions[0]['keypoints']
            
            for mm in maps:
                N, W, H = mm.shape
                
                mm_s = torch.nn.functional.softmax(mm.view(N, -1), dim = 1)
                mm_s = mm_s.view(N, W, H)
                
                mm_s = mm_s.cpu().detach().numpy()
                
                plt.figure()
                plt.imshow(mm_s.max(0))
        
        
        #break
        