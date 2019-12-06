o#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.models import CPM_PAF, PretrainedBackBone, get_preeval_func
from worm_poses.flow import SkelMapsSimpleFlow
from worm_poses.encoders import extrack_keypoints, join_skeletons_simple, get_best_match, resample_curve

import tqdm
import torch
import matplotlib.pylab as plt
import numpy as np

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
                             backbone = backbone,
                              is_PAF = False
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

#%%
if __name__ == '__main__':
    
    bn = 'v1_CPM_maxlikelihood_20190624_213353_adam_lr0.0001_wd0.0_batch96'
    #bn = 'v1-fixW_CPM_mse_20190624_230012_adam_lr0.001_wd0.0_batch96'
    #bn = 'v1_CPM_mse_20190624_231204_adam_lr0.0001_wd0.0_batch96'
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/v1' / bn / 'checkpoint.pth.tar'
    
    
    cuda_id = 0
    
    device = get_device(cuda_id)
    model, preeval_func = load_model(model_path)
    model = model.to(device)
    
    
    
    roi_size = 96
    set2read = 'validation'
    #root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190606_174815/'

    root_dir ='/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    root_dir = Path(root_dir)
    gen = SkelMapsSimpleFlow(root_dir = root_dir,
                              set2read = set2read,
                              roi_size = roi_size, 
                              data_types = ['manual'],
                              return_raw_skels = True
                              )
    
    #%%
    cutoff_error = 3.
    all_errors = []
    for w_ind, (X,Y) in enumerate(tqdm.tqdm(gen)):
        
        skels_true = Y[-1]
        
        X = torch.tensor(X[None])
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            
            cpm_maps_r = outs[2]
            cpm_maps_r = preeval_func(cpm_maps_r)
            
        
        cpm_map = cpm_maps_r[0].detach().cpu().numpy()
        #paf_maps = paf_maps_r[0].detach().cpu().numpy()
        
        keypoints = extrack_keypoints(cpm_map, threshold_relative = 0.25, threshold_abs = .25)
        skels_pred = join_skeletons_simple(keypoints, max_edge_dist = 10)
        
        skels_pred = [resample_curve(x, skels_true[0].shape[0]) for x in skels_pred]
        
        
        closest_ind, closest_error = get_best_match(skels_pred, skels_true)
        
        
        is_valid = len(set(closest_ind)) == len(skels_true) #all the skeletons where selected
        all_errors.append((is_valid, closest_error))
        
        is_valid = is_valid & all([x < cutoff_error for x in closest_error])# ... and all average distance between skeletons is less than 1
        
       
        #if ii % 10 == 0:
        if not is_valid:
            #%%
            roi = X[0, 0]
            fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(10, 5))
            axs[0][0].imshow(cpm_map.max(axis=0))
            axs[0][1].imshow(cpm_map[3])
            axs[0][2].imshow(cpm_map[-1])
            
            axs[1][1].imshow(roi, cmap='gray')
            for ss in skels_pred:
                axs[1][1].plot(ss[:, 0], ss[:, 1], '.-')
                axs[1][1].plot(ss[25, 0], ss[25, 1], 'o')
            axs[1][1].set_title('predictions')
                
            axs[1][0].imshow(roi, cmap='gray')
            for ss in skels_true:
                axs[1][0].plot(ss[:, 0], ss[:, 1], '.-')
                axs[1][0].plot(ss[25, 0], ss[25, 1], 'o')
            axs[1][0].set_title('true')
            plt.suptitle(w_ind)
            
            
    #%%
    valid_n_worms, roi_errors = zip(*all_errors)
    valid_n_errors = [np.max(x) < cutoff_error if len(x) else False for x in roi_errors]
    
    roi_valids = np.array(valid_n_worms) & np.array(valid_n_errors)
    print(np.mean(roi_valids))
    
