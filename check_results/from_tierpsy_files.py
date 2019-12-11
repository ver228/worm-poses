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
from worm_poses.utils import get_device
from worm_poses.models import PoseDetector
import tqdm

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    
    bn = 'v2_openpose_maxlikelihood_20191210_172128_adam_lr0.0001_wd0.0_batch14'
    #bn = 'v2_openpose+light_maxlikelihood_20191210_172128_adam_lr0.0001_wd0.0_batch32'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    #'checkpoint.pth.tar'
    
    
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    
    if 'openpose+light' in bn:
        model_args = dict(
            n_segments = 9,
            n_affinity_maps = 8,
            features_type = 'vgg11',
            n_stages = 4,
            )
    else:
        model_args = dict(
            n_segments = 25,
            n_affinity_maps = 21,
            features_type = 'vgg19',
            n_stages = 6,
            )
    
    model = PoseDetector(**model_args, return_belive_maps = True)
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    fnames = ['/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5',
              '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
              ]
    for mask_file in tqdm.tqdm(fnames):
        
        mask_file = Path(mask_file)
        with tables.File(mask_file, 'r') as fid:
            img = fid.get_node('/full_data')[-1]
        
        img = img.astype(np.float32)/255.
        
        X = torch.from_numpy(img).float()[None, None]
        predictions, belive_maps = model(X)
        
        
        # skels = predictions[0]['skeletons'].detach().cpu().numpy()
        
        # fig, ax = plt.subplots(1,1)
        # ax.imshow(img, cmap='gray')
        # plt.plot(skels[:, 0], skels[:, 1], 'r.')
        #%%
        
        cpm, paf = belive_maps
        cpm = cpm[0].detach().numpy()
        cpm_max = cpm.max(axis = 0)
        
        paf = paf[0].detach().numpy()
        paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
        #%%
        fig, axs = plt.subplots(1,5, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(cpm[0])
        axs[2].imshow(np.linalg.norm(paf[0], axis = 0))
        axs[3].imshow(cpm[-1])
        axs[4].imshow(np.linalg.norm(paf[-1], axis = 0))
        #axs[3].imshow(cpm_max)
        #axs[4].imshow(paf_max)
        
        
        
        
        
        
        