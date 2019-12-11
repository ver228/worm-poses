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

import tables
import torch
from worm_poses.trainer import get_device

from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    from matplotlib import patches
    
    #bn = 'detection-singles_fasterrcnn_20190628_174616_adam_lr0.0001_wd0.0_batch16'
    
    bn = 'v2_openpose_maxlikelihood_20191209_093737_adam_lr0.0001_wd0.0_batch20'
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'checkpoint.pth.tar'
    
    
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    roi_size = 2048
    if 'keypointrcnn' in bn:
        model = keypointrcnn_resnet50_fpn(
                                        num_classes = 2, 
                                        num_keypoints = 25,
                                        min_size = roi_size,
                                        max_size = roi_size,
                                        image_mean = [0., 0., 0.],
                                        image_std = [1., 1., 1.], 
                                        pretrained = False
                                        )
    elif 'fasterrcnn' in bn:
        model = fasterrcnn_resnet50_fpn(
                                    num_classes = 2, 
                                    min_size = roi_size,
                                    max_size = roi_size,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.],
                                    pretrained = False
                                    )
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fnames = ['/Users/avelinojaver/Downloads/recording61.2r_X1.hdf5']
    
    #fnames = ['/Users/avelinojaver/Imperial College London/Feriani, Luigi - Mating_for_Avelino/wildMating/MaskedVideos/20180817_wildMating/wildMating3.2_MY23_cross_MY23_cross_PC1_Ch2_17082018_123407.hdf5']
    #fnames = ['/Users/avelinojaver/Imperial College London/Feriani, Luigi - Mating_for_Avelino/Mating_Assay/Mating_Assay_030718/MaskedVideos/Set1_CB369_CB1490/Set1_CB369_CB1490_Ch1_03072018_163429.hdf5']
    #fnames = ['/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5']
    
    
    fnames = ['/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5',
              '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5']
    for mask_file in fnames:
        mask_file = Path(mask_file)
        with tables.File(mask_file, 'r') as fid:
            img = fid.get_node('/full_data')[-1]
        
        img = img.astype(np.float32)/255.
        
        X = torch.from_numpy(np.repeat(img[None], 3, axis=0)).float()[None]
        predictions = model(X)
        
        
        #%%
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap='gray')
        
        preds = predictions[0]
        for lab, bbox in zip(preds['labels'], preds['boxes']):
            bbox = bbox.detach().numpy()
            
            xmin, ymin, xmax, ymax = bbox
            ww = xmax - xmin + 1
            hh = ymax - ymin + 1
            rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            s = str(lab.item())
            ax.text(x, y, s, color='cyan', fontsize=12)
            #%%
            
            
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
                