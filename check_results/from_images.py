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

import cv2
import torch
from worm_poses.models import PoseDetector
import tqdm

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
    
    bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    
    
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
    model = PoseDetector(**model_args, return_belive_maps = True)
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC010_v1_images/')
    
    
    fnames = list(root_dir.glob('*w2*.tif'))[:10]
    for ifname, fname in enumerate(tqdm.tqdm(fnames)):
        img = cv2.imread(str(fname), -1)
        #img = img.astype(np.float32)/4095.
        img = img.astype(np.float32)/img.max()
        
        X = torch.from_numpy(img).float()[None, None]
        predictions, belive_maps = model(X)
        
        
        #skels = predictions['skeletons'].detach().cpu().numpy()
        skels = predictions[0]['skeletons'].detach().cpu().numpy()
        
        cpm, paf = belive_maps
        cpm = cpm[0].detach().numpy()
        cpm_max = cpm.max(axis = 0)
        
        paf = paf[0].detach().numpy()
        paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap='gray')
        for ii in range(model.n_segments):
            valid = skels[..., 0] == ii
            plt.plot(skels[valid, -2], skels[valid, -1], '.')
        
        
        fig, axs = plt.subplots(1,5, figsize = (30, 5), sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(cpm[3])
        axs[2].imshow(np.linalg.norm(paf[3], axis = 0))
        #axs[3].imshow(cpm[-3])
        #axs[4].imshow(np.linalg.norm(paf[-3], axis = 0))
        axs[3].imshow(cpm[-1])
        axs[4].imshow(np.linalg.norm(paf[-1], axis = 0))
        #axs[3].imshow(cpm_max)
        #axs[4].imshow(paf_max)
        
        #%%
        # PAF = belive_maps[-1]
        
        # n_affinity_maps = 8
        # n_segments = 8
        # PAF_seg_dist = 1
        
        # skeletons = predictions['skeletons']
        
        # segment_id = skeletons[:, 1]
        # skeletons_bxy = skeletons[:, (0, 2, 3)]
        
        # skels_by_segments = []
        # skels_inds = torch.arange(len(skeletons), device = PAF.device)
        # for ii in range(n_segments):
        #     valid = segment_id == ii
        #     skel_l = skels_inds[valid]
        #     skels = skeletons_bxy[valid]
        #     skels_by_segments.append((skel_l, skels))
        
        # #%%
        
        # points_grouped = []
        # for ii in range(n_segments):
        #     good = seg_inds == ii
        #     points_grouped.append((skels_inds[good], skels_xy[good]))
            
            
        # edges = []
        # for i1 in range(n_affinity_maps):
            
        #     if i1 < n_affinity_maps - 1:
        #         i2 = i1 + PAF_seg_dist
        #     else:
        #         i2 = i1 - max(1, PAF_seg_dist//2)
            
        #     p1_ind, p1_l = points_grouped[i1]
        #     p2_ind, p2_l = points_grouped[i2]
            
        #     n_p1 = len(p1_l)
        #     n_p2 = len(p2_l)
            
        #     p1 = p1_l[None].repeat(n_p2, 1,  1)
        #     p2 = p2_l[:, None].repeat(1, n_p1, 1)
        #     midpoints = (p1 + p2)/2
            
        #     inds = torch.stack((p1, p2, midpoints))
        #     paf_vals = PAF[i1][:, inds[..., 1], inds[..., 0]] 
            
        #     p1_f, p2_f = p1.float(), p2.float()
            
        #     target_v = (p2_f - p1_f)
        #     R = (target_v**2).sum(dim=2).sqrt()
        #     target_v.div_(R.unsqueeze(2))
        #     target_v = target_v.permute((2, 0, 1))
            
        #     line_integral = (target_v.unsqueeze(1)*paf_vals).sum(dim=0).mean(dim=0)
            
        #     pairs = torch.nonzero(R < self.keypoint_max_dist, as_tuple=True)
        #     paf_vals = line_integral[pairs[0], pairs[1]]
        #     paf_vals[torch.isnan(paf_vals)] = 0
            
        #     R_vals = R[pairs[0], pairs[1]]
            
        #     points_pairs = torch.stack((p1_ind[pairs[1]], p2_ind[pairs[0]]))
        #     costs = torch.stack((paf_vals, R_vals))
            
        #     edges.append((points_pairs, costs))
        
        #%%
        # n_segments, _ = skeletons.shape

        # n_affinity_maps, _, W, H = PAF.shape
        
        # seg_inds = skeletons[:, 0]
        # skels_xy = skeletons[:, 1:]
        
        # skels_inds = torch.arange(len(skeletons), device = PAF.device)
        
        # points_grouped = []
        # for ii in range(n_segments):
        #     good = seg_inds == ii
        #     points_grouped.append((skels_inds[good], skels_xy[good]))
        
        # edges = []
        # for i1 in range(n_affinity_maps):
            
        #     if i1 < n_affinity_maps - 1:
        #         i2 = i1 + self.PAF_seg_dist
        #     else:
        #         i2 = i1 - max(1, self.PAF_seg_dist//2)
            
        #     p1_ind, p1_l = points_grouped[i1]
        #     p2_ind, p2_l = points_grouped[i2]
            
        #     n_p1 = len(p1_l)
        #     n_p2 = len(p2_l)
            
        #     p1 = p1_l[None].repeat(n_p2, 1,  1)
        #     p2 = p2_l[:, None].repeat(1, n_p1, 1)
        #     midpoints = (p1 + p2)/2
            
        #     inds = torch.stack((p1, p2, midpoints))
        #     paf_vals = PAF[i1][:, inds[..., 1], inds[..., 0]] 
            
        #     p1_f, p2_f = p1.float(), p2.float()
            
        #     target_v = (p2_f - p1_f)
        #     R = (target_v**2).sum(dim=2).sqrt()
        #     target_v.div_(R.unsqueeze(2))
        #     target_v = target_v.permute((2, 0, 1))
            
        #     line_integral = (target_v.unsqueeze(1)*paf_vals).sum(dim=0).mean(dim=0)
            
        #     pairs = torch.nonzero(R < self.keypoint_max_dist, as_tuple=True)
        #     paf_vals = line_integral[pairs[0], pairs[1]]
        #     paf_vals[torch.isnan(paf_vals)] = 0
            
        #     R_vals = R[pairs[0], pairs[1]]
            
        #     points_pairs = torch.stack((p1_ind[pairs[1]], p2_ind[pairs[0]]))
        #     costs = torch.stack((paf_vals, R_vals))
            
        #     edges.append((points_pairs, costs))
            
        # edges_indices, edges_costs = zip(*edges)    
        
        # #%%
        # skeletons = predictions[0]['skeletons']
        # edges_indices = predictions[0]['edges_indices']
        # edges_costs = predictions[0]['edges_costs']
        # #% greedy linking
        
        # match_min_val = -40
        # skels_linked = []
        # for dat in list(zip(edges_indices, edges_costs)):
        #     (points_pairs, costs) = [x.cpu().detach().numpy() for x in dat]
            
        #     grouped_pairs = {}
        #     for (p1, p2), (W, R) in zip(points_pairs.T, costs.T):
        #         if not p1 in grouped_pairs:
        #             grouped_pairs[p1] = []
        #         grouped_pairs[p1].append((p2, W, R))
            
            
        #     def _append_best_match(key, list2append = []):
        #         if key in grouped_pairs:
        #             possible_matches = grouped_pairs[key]
        #             best_match = max(possible_matches, key = lambda x : x[1])
                    
        #             if best_match[1] >= match_min_val:
        #                 matched_ind = best_match[0]
        #                 assigned_.append(matched_ind)
        #                 list2append.append(matched_ind)
                
        #         return list2append
                
                
                
        #     assigned_ = []
        #     for s in skels_linked:
        #         k = s[-1]
        #         s = _append_best_match(k, s)
        #         if s[-1] != k:
                
        #             assigned_.append(k)
            
        #     left_dat = set(grouped_pairs.keys()) - set(assigned_)
            
        #     dat2add = [_append_best_match(k, [])for k in left_dat]
        #     skels_linked += [x for x in dat2add if x]
        
        
        # skels = skeletons.detach().numpy()
        
        
        # plt.figure()
        # plt.imshow(img, cmap='gray')
        # for ind in skels_linked:
            
        #     x = skels[ind, 1]
        #     y = skels[ind, 2]
            
            
        #     plt.plot(x, y, '.-')
        
            
            