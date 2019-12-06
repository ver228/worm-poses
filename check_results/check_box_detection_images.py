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

import tqdm
import cv2
    
import torch
from worm_poses.trainer import get_device
from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    from matplotlib import patches
    
    bn = 'detection-singles_keypointrcnn_20190703_085950_adam_lr0.0001_wd0.0_batch8'
    #bn = 'detection-singles_fasterrcnn_20190628_174616_adam_lr0.0001_wd0.0_batch16'
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/detection-singles'  / bn / 'checkpoint.pth.tar'
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    min_size = 520
    max_size = 696
    if 'keypointrcnn' in bn:
        model = keypointrcnn_resnet50_fpn(
                                        num_classes = 2, 
                                        num_keypoints = 25,
                                        min_size = min_size,
                                        max_size = max_size,
                                        image_mean = [0., 0., 0.],
                                        image_std = [1., 1., 1.], 
                                        pretrained = False
                                        )
    elif 'fasterrcnn' in bn:
        model = fasterrcnn_resnet50_fpn(
                                    num_classes = 2, 
                                    min_size = min_size,
                                    max_size = max_size,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.],
                                    pretrained = False
                                    )
    
   
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC010_v1_images/')
    
    for ifname, fname in enumerate(tqdm.tqdm(list(root_dir.glob('*w2*.tif')))):
        img = cv2.imread(str(fname), -1)
        img = img.astype(np.float32)/4095.
        
        X = torch.from_numpy(np.repeat(img[None], 3, axis=0)).float()
        predictions = model([X])
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap='gray')
        
        preds = predictions[0]
        for bbox in preds['boxes']:
            bbox = bbox.detach().numpy()
            
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        
        if 'keypoints' in preds:
            maps = preds['keypoints']
            bboxes = preds['boxes']
            for mm, bbox in zip(maps, bboxes):
                N, W, H = mm.shape
                
                mm_s = torch.nn.functional.softmax(mm.view(N, -1), dim = 1)
                mm_s = mm_s.view(N, W, H)
                
                mm_s = mm_s.cpu().detach().numpy()
                
                mm_max = mm_s.max(0)
                xmin, ymin, xmax, ymax = bbox.round().int().numpy()
                roi = img[ ymin:ymax, xmin:xmax]
                
                
                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                axs[0].imshow(roi)
                axs[1].imshow(mm_max)
        
        if ifname > 2:
            break