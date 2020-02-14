#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
_script_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(_script_dir))

from worm_poses.models import PoseDetector

import cv2
import torch
import numpy as np
import random
import tqdm
import gzip
import pickle

def link_segments(preds, 
                n_segments = 8,
                min_PAF = 0.25,
                is_skel_half = True
                ):
    #%%
    edges_cost = preds['edges_costs']
    edges_indeces = preds['edges_indices']
    points = preds['skeletons']
    
    #get the best matches per point
    PAF_cost = edges_cost[0]
    valid = PAF_cost >= min_PAF
    PAF_cost = PAF_cost[valid]
    edges_indeces = edges_indeces[:, valid]
    
    inds = np.argsort(PAF_cost)[::-1]
    edges_indeces = edges_indeces[:, inds]
    _, valid_index =  np.unique(edges_indeces[0], return_index = True )
    best_matches = {x[0]:x[1] for x in edges_indeces[:, valid_index].T}
    matched_points = set(best_matches.keys())
    
    assert (edges_indeces < len(points)).all()
    points =  np.concatenate((np.arange(len(points))[:, None], points), axis=1)     
    
    
    #%%
    #add the point_id column 
    segments_linked = []
    for ipoint in range(n_segments):
        points_in_segment = points[points[:, 1] == ipoint]
        
        prev_inds = {x[-1][0]:x for x in segments_linked}
        
        matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
        matched_indices_cur = [best_matches[x] for x in matched_indices_prev]
    
        new_points = set(points_in_segment[:, 0]) - set(matched_indices_cur)
    
        for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
            prev_inds[k1].append(points[k2])
        
        segments_linked += [[points[x]] for x in new_points]
        
        #%%
    return segments_linked

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    
    #bn = 'v3_openpose+light+head_maxlikelihood_20200118_100732_adam_lr0.0001_wd0.0_batch24'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200129_080810_adam_lr0.0001_wd0.0_batch24'
    #bn = 'v4_openpose+light+head_maxlikelihood_20200129_104454_adam_lr0.0001_wd0.0_batch24'
    #bn = 'v4PAFflat_R+openpose+light+head_maxlikelihood_20200206_105310_adam_lr1e-05_wd0.0_batch24'
    #bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
        
    #bn = 'v5_openpose+head_maxlikelihood_20200212_162013_adam_lr0.0001_wd0.0_batch14'
    #bn = 'v5_openpose+light+head_maxlikelihood_20200212_161540_adam_lr0.0001_wd0.0_batch24'
    
    #bn = 'v5mixup_R+openpose+light+head_maxlikelihood_20200213_180515_adam_lr0.0001_wd0.0_batch24'
    bn = 'v5mixup_R+openpose+head_maxlikelihood_20200214_064610_adam_lr0.0001_wd0.0_batch14'
    #bn = 'v5mixup_openpose+light+head_maxlikelihood_20200213_180448_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    #model_path = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/models/' / bn / 'model_best.pth.tar'
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
            n_affinity_maps = 7,#8,
            features_type = 'vgg11',
            n_stages = 4,
            )
        
      
    else:
        model_args = dict(
            n_segments = 8,
            n_affinity_maps = 7,#8,
            features_type = 'vgg19',
            n_stages = 6,
            )
        
    if '+head' in bn:
        model_args['n_segments'] += 1
        
    #%%
    model = PoseDetector(**model_args, return_belive_maps = True, keypoint_max_dist = 100, nms_min_distance = 3)
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    src_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training_filtered/manual_test.p.zip'
    #src_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training_filtered/from_tierpsy_test.p.zip'
    #src_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training_filtered/from_NNv1_test.p.zip'
    
    with gzip.GzipFile(src_file, 'rb') as fid:
        data_raw = pickle.load(fid)
    #%%
    model.nms_min_distance = 1
    n_segments = 8
    
    #dat = data_raw[-50:]
    
    dat = [data_raw[ii] for ii in [722, 721, 719, 717, 49, 48, 44, 41, 21, 17, 10]]
    for ii, out in enumerate(tqdm.tqdm(dat)): #enumerate():
        roi = out[1] if out[1] is not None else out[0]
        skels = out[3]
        if skels.shape[0] < 1:
            continue
        
        X = roi.astype(np.float32)/255
        X = torch.from_numpy(X).float()[None, None]
        
        
        #scale_factor = 1
        #X = nn.functional.interpolate(X, scale_factor =scale_factor)
        predictions, belive_maps = model(X)
        predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
        
        segments = link_segments(predictions[0])
        segments = [np.stack(x) for x in segments]
        
        p = predictions[0]['skeletons']
        p = p[p[:, 0] < n_segments]
        
        scores_abs = predictions[0]['scores_abs']
        
        fig, axs = plt.subplots(2, 4, figsize = (20, 20), sharex = True, sharey = True)
        for ax in axs.flatten():
            ax.imshow(roi, cmap = 'gray')
            ax.axis('off')
            
        for skel in skels:
            axs[0][1].plot(skel[:, 0], skel[:, 1], '.')
        
        
        axs[0][2].scatter(p[:, -2], p[:, -1],  c = p[:, 0], cmap = 'jet')
        for seg in segments:
            delx = (random.random()-0.5)*1e-0
            dely = (random.random()-0.5)*1e-0
            axs[0][3].plot(seg[:, -2] + delx, seg[:, -1] + dely, '.-')
        
        cpm, paf = belive_maps
        cpm = cpm[0].detach().numpy()
        cpm_max = cpm[:n_segments].max(axis = 0)
        
        paf = paf[0].detach().numpy()
        paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
        
        axs[1][0].imshow(cpm_max)
        axs[1][1].imshow(paf_max)
        axs[1][2].imshow(np.linalg.norm(paf[0], axis = 0))
        axs[1][3].imshow(np.linalg.norm(paf[-1], axis = 0))

        plt.suptitle(bn)

        #%%
        # break
        