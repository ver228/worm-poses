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

from from_validation import link_segments

import cv2
import torch
from worm_poses.models import PoseDetector
import tqdm
import torch.nn.functional as F

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    
    #bn = 'v2_openpose+light_maxlikelihood_20191210_172128_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    
    #bn = 'v2_openpose+head_maxlikelihood_20191219_150412_adam_lr0.0001_wd0.0_batch20'
    #bn = 'v2_openpose+light+full_maxlikelihood_20191226_114337_adam_lr0.0001_wd0.0_batch32'
    
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191228_165156_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_094906_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    
    #bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200129_080810_adam_lr0.0001_wd0.0_batch24'
    
    #bn = 'v4_openpose+light+head_maxlikelihood_20200129_104454_adam_lr0.0001_wd0.0_batch24'
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    
    
    #model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    model_path = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/models/' / bn / 'model_best.pth.tar'
    #'checkpoint.pth.tar'
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    if 'openpose+light+fullsym' in bn:
        model_args = dict(
            n_segments = 15,
            n_affinity_maps = 14,
            features_type = 'vgg11',
            n_stages = 4,
            )
        
    elif 'openpose+light' in bn:
        model_args = dict(
            n_segments = 8,
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
        
    if '+head' in bn:
        model_args['n_segments'] += 1
        
    #%%
    model = PoseDetector(**model_args, return_belive_maps = True, keypoint_max_dist = 20, nms_min_distance = 5)
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC010_v1_images/')
    
    
    fnames = list(root_dir.glob('*w2*.tif'))[:10]
    for ifname, fname in enumerate(tqdm.tqdm(fnames)):
        
        img = cv2.imread(str(fname), -1)
        #img = img.astype(np.float32)/4095.
        img = img.astype(np.float32)/img.max()
        #img = cv2.blur(img, ksize = (5,5))
        
        X = torch.from_numpy(img).float()[None, None]
        
        scale_factor = 1.#0.5
        X = F.interpolate(X, scale_factor = scale_factor)
        
        predictions, belive_maps = model(X)
        predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
        
        segments = link_segments(predictions[0])
        segments = [np.stack(x) for x in segments]
        
        #skels = predictions['skeletons'].detach().cpu().numpy()
        skels = predictions[0]['skeletons'].astype(np.float32)
        
        skels[:, -2:] /= scale_factor 
        
        cpm, paf = belive_maps
        cpm = cpm[0].detach().numpy()
        cpm_max = cpm.max(axis = 0)
        
        paf = paf[0].detach().numpy()
        paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
        
        fig, axs = plt.subplots(1,3, figsize = (30, 10), sharex = True, sharey = True)
        for ax in axs:
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
        for ii in range(model.n_segments):
            valid = skels[..., 0] == ii
            axs[1].plot(skels[valid, -2], skels[valid, -1], '.')
        
        for seg in segments:
            seg = seg[:, -2:]/ scale_factor
            axs[2].plot(seg[:, -2], seg[:, -1], '-')
            #delx = (random.random()-0.5)*1e-0
            #dely = (random.random()-0.5)*1e-0
            #axs[2].plot(seg[:, -2] + delx, seg[:, -1] + dely, '-')
        #break
            
        
        #%%
        # fig, axs = plt.subplots(1,5, figsize = (30, 5), sharex = True, sharey = True)
        # axs[0].imshow(img, cmap = 'gray')
        # axs[1].imshow(cpm[3])
        # axs[2].imshow(np.linalg.norm(paf[3], axis = 0))
        # #axs[3].imshow(cpm[-3])
        # #axs[4].imshow(np.linalg.norm(paf[-3], axis = 0))
        # axs[3].imshow(cpm[-1])
        # axs[4].imshow(np.linalg.norm(paf[-1], axis = 0))
        # #axs[3].imshow(cpm_max)
        # #axs[4].imshow(paf_max)
        
        