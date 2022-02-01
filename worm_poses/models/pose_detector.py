#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:20:23 2019

@author: avelinojaver
"""

from .openpose import OpenPoseCPM
from .losses import MaximumLikelihoodLoss, LossWithBeliveMaps, SymetricMLELoss, SymetricPAFLoss


import torch
from torch import nn
import torch.nn.functional as F


def _normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = F.softmax(hh, dim = 2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def get_loc_loss(loss_type):
    '''
    Parser of the loss type to be used.

    We can use either a maxlikelihood loss that maximizes the probability of the pixels with segments,
    or we could generate a reference map by convolving the localization points with a gaussian and 
    calculating the MSE.

    TODO -> this is a bit complex, since we are only using maxlikelihood we probably remove the rest.

    '''
    criterion = None
    if loss_type == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        preevaluation = _normalize_softmax
    
    elif loss_type == 'maxlikelihood+symetric':
        criterion = SymetricMLELoss()
        preevaluation = _normalize_softmax
        
    
    else:
        parts = loss_type.split('-')
        loss_t = parts[0]
        
        gauss_sigma = [x for x in parts if x[0] == 'G'] #this must exists
        gauss_sigma = float(gauss_sigma[0][1:])
         
        increase_factor = [x for x in parts if x[0] == 'F'] #this one is optional
        increase_factor = float(increase_factor[0][1:]) if increase_factor else 1.
        
        
        is_regularized = [x for x in parts if x == 'reg']
        is_regularized = len(is_regularized) > 0
        
        
        preevaluation = lambda x : x
        if loss_t == 'l1smooth':
            target_loss = nn.SmoothL1Loss()
        elif loss_t == 'l2':
            target_loss = nn.MSELoss()
        elif loss_t == 'l1':
            target_loss = nn.L1Loss()
            
        criterion = LossWithBeliveMaps(target_loss, 
                                       gauss_sigma = gauss_sigma, 
                                       is_regularized = is_regularized,
                                       increase_factor = increase_factor
                                       )    
    if criterion is None:
        raise ValueError(loss_type)
    return criterion, preevaluation

class BeliveMapsNMS(nn.Module):
    def __init__(self, 
                 threshold_abs = 0.0, 
                 threshold_rel = None, 
                 min_distance = 3,
                 max_num_peaks = 100
                 
                 ):
        super().__init__()
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.min_distance = min_distance
        self.max_num_peaks = max_num_peaks
        
    def forward(self, belive_map):
        B,S,H,W = belive_map.shape
        
        kernel_size = 2 * self.min_distance + 1
        
        n_batch, n_channels, w, h = belive_map.shape
        hh = belive_map.contiguous().view(n_batch, n_channels, -1)
        max_vals, _ = hh.max(dim=2)
        
        x_max = F.max_pool2d(belive_map, kernel_size, stride = 1, padding = kernel_size//2)
        x_mask = (x_max == belive_map) #nms using local maxima filtering
        
        
        threshold_abs = self.threshold_abs
        if threshold_abs == 0:
            threshold_abs = 1/(H*W)*2
        
        x_mask &= (belive_map > threshold_abs) 
        if self.threshold_rel is not None:
            vmax = max_vals.view(n_batch, n_channels, 1, 1)
            x_mask &= (belive_map > self.threshold_rel*vmax)
        
            
        outputs = []
        
        for xi, xm, xmax in zip(belive_map, x_mask, max_vals):
            ind = xm.nonzero()
            
            skeletons = ind[:, [0, 2, 1]]
            scores_abs = xi[ind[:, 0], ind[:, 1], ind[:, 2]]
            
            if ind.shape[0] > self.max_num_peaks*S:
                #too many peaks, I need to reduce the size...
                scores_abs_l = []
                skeletons_l = []
                for n_seg in range(S):
                    valid = ind[:, 0] == n_seg
                    _scores = scores_abs[valid]
                    n_valid = len(_scores)
                    if n_valid:
                        vals, iis = torch.topk(_scores, min(self.max_num_peaks, n_valid))
                        scores_abs_l.append(vals)
                        skeletons_l.append(skeletons[valid][iis])
                scores_abs = torch.cat(scores_abs_l)
                skeletons = torch.cat(skeletons_l)
                
            outputs.append((skeletons, scores_abs))
        
        return outputs

class PAFWeighter(nn.Module):
    def __init__(self, keypoint_max_dist = 20, n_segments = 8, PAF_seg_dist = 1, n_points_integral = 10):
        super().__init__()
        self.n_segments = n_segments
        self.keypoint_max_dist = keypoint_max_dist
        self.PAF_seg_dist = PAF_seg_dist
        self.n_points_integral = n_points_integral
        #self.factors = np.linspace(0, 1, num = self.n_points_integral)
        
        
    def forward(self, skeletons, PAF):
        if not len(skeletons):
            edges_indices = torch.zeros((0, 2), dtype = skeletons.dtype, device = skeletons.device)
            edges_costs = torch.zeros((0, 2), dtype = torch.float, device = skeletons.device)
            return edges_indices, edges_costs
        
        
        n_affinity_maps, _, W, H = PAF.shape
        
        seg_inds = skeletons[:, 0]
        skels_xy = skeletons[:, 1:]
        
        skels_inds = torch.arange(len(skeletons), device = PAF.device)
        
        points_grouped = []
        for ii in range(self.n_segments):
            good = seg_inds == ii
            points_grouped.append((skels_inds[good], skels_xy[good]))
        
        edges = []
        
        N = self.n_points_integral - 1 
        for i1 in range(n_affinity_maps):
            i2 = i1 + self.PAF_seg_dist
            # if i1 < n_affinity_maps - 1:
            #     i2 = i1 + self.PAF_seg_dist
            # else:
            #     i2 = i1 - max(1, self.PAF_seg_dist//2)
            
            p1_ind, p1_l = points_grouped[i1]
            p2_ind, p2_l = points_grouped[i2]
            
            n_p1 = len(p1_l)
            n_p2 = len(p2_l)
            
            p1 = p1_l[None].repeat(n_p2, 1,  1)
            p2 = p2_l[:, None].repeat(1, n_p1, 1)
            
            inds = [(p1*(N-x) + p2*x)//N for x in range(self.n_points_integral)]
            inds = torch.stack(inds)
            #midpoints = (p1 + p2)/2
            #inds = torch.stack([p1, p2] +  midpoints))
            paf_vals = PAF[i1][:, inds[..., 1], inds[..., 0]] 
            
            p1_f, p2_f = p1.float(), p2.float()
            
            target_v = (p2_f - p1_f)
            R = (target_v**2).sum(dim=2).sqrt()
            target_v.div_(R.unsqueeze(2))
            target_v = target_v.permute((2, 0, 1))
            
            line_integral = (target_v.unsqueeze(1)*paf_vals).sum(dim=0).mean(dim=0)
            
            pairs = torch.nonzero(R < self.keypoint_max_dist, as_tuple=True)
            paf_vals = line_integral[pairs[0], pairs[1]]
            paf_vals[torch.isnan(paf_vals)] = 0
            
            R_vals = R[pairs[0], pairs[1]]
            
            points_pairs = torch.stack((p1_ind[pairs[1]], p2_ind[pairs[0]]))
            costs = torch.stack((paf_vals, R_vals))
            
            edges.append((points_pairs, costs))
         
        edges_indices, edges_costs =  zip(*edges)
        edges_indices = torch.cat(edges_indices, dim=1)
        edges_costs = torch.cat(edges_costs, dim=1)
        
        
        return edges_indices, edges_costs

    
class PoseDetector(nn.Module):
    def __init__(self, 
                 pose_loss_type = 'maxlikelihood',
                 
                 n_inputs = 1,
                 n_stages = 6, 
                 n_segments = 25,
                 n_affinity_maps = 20,
                 features_type = 'vgg19',
                 
                 nms_threshold_abs = 0,
                 nms_threshold_rel = 0.01,
                 nms_min_distance = 1,
                 keypoint_max_dist = 20,
                 max_poses = 100,
                 
                 use_head_loss = False,
                 return_belive_maps = False,
                 
                 ):
        
        
        super().__init__()
        
        _dum = set(dir(self))
        self.nms_threshold_abs = nms_threshold_abs
        self.nms_threshold_rel = nms_threshold_rel
        self.nms_min_distance = nms_min_distance
        self.max_poses = max_poses
        self.keypoint_max_dist = keypoint_max_dist
        
        self.pose_loss_type = pose_loss_type
        self.n_segments = n_segments
        self.n_stages = n_stages
        self.n_affinity_maps = n_affinity_maps
        self.features_type = features_type
        
        self.use_head_loss = use_head_loss
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        n_outs = n_segments + 1 if use_head_loss else n_segments
        
        self.mapping_network = OpenPoseCPM(n_inputs = n_inputs,
                             n_stages = n_stages, 
                             n_segments = n_outs,
                             n_affinity_maps = n_affinity_maps,
                             features_type = features_type
                             )
        
        self.cpm_criterion, self.preevaluation = get_loc_loss(pose_loss_type)
        self.paf_criterion = SymetricPAFLoss()  if 'symetric' in pose_loss_type else nn.MSELoss()
        
        self.nms = BeliveMapsNMS(nms_threshold_abs, 
                                 nms_threshold_rel, 
                                 nms_min_distance,
                                 max_poses)
        self.paf_weighter = PAFWeighter(keypoint_max_dist = keypoint_max_dist, n_segments = n_segments)
        
        self.return_belive_maps = return_belive_maps
        
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    
    def forward(self, x, targets = None):
        pose_map, PAFs = self.mapping_network(x)
        
        outputs = []
        
        if targets is not None:
            skel_targets = [t['skels'] for t in targets]
            skel_maps = pose_map[:, :self.n_segments]
            
            if 'symetric' in self.pose_loss_type:
                is_valid_ht_targets = [t['is_valid_ht'] for t in targets]
                cpm_loss = self.cpm_criterion(skel_maps, skel_targets, is_valid_ht = is_valid_ht_targets)
            else:
                cpm_loss = self.cpm_criterion(skel_maps, skel_targets)
            
            target_PAFs = torch.stack([t['PAF'] for t in targets])
            if self.training:
                paf_loss = sum([self.paf_criterion(x, target_PAFs) for x in PAFs])
            else:
                paf_loss = self.paf_criterion(PAFs, target_PAFs)
            
            loss = dict(
                cpm_loss = cpm_loss,
                paf_loss = paf_loss
                )
            
            if self.use_head_loss:
                head_targets = [t['heads'].unsqueeze(1) for t in targets if t['heads'].shape[0]>0]
                
                _valid = [t['heads'].shape[0]>0 for t in targets]
                cpm_heads = pose_map[_valid, self.n_segments:]
                
                assert cpm_heads.shape[0] == len(head_targets)
                if head_targets:
                    loss['cpm_head_loss'] = self.cpm_criterion(cpm_heads, head_targets)
            
            outputs.append(loss)
        
        if not self.training:
            xhat = self.preevaluation(pose_map.detach())
            
            outs = self.nms(xhat.detach())
            
            result = []
            for (skeletons, scores_abs), PAF in zip(outs, PAFs):
                edges_indices, edges_costs = self.paf_weighter(skeletons, PAF)
                #edges_indices, edges_costs = [], [] 
                
                if all(edges_indices.size()) and (edges_indices.max() > skeletons.shape[0]):
                    import pdb
                    pdb.set_trace()
                
                
                result.append(
                    dict(
                        skeletons =  skeletons,
                        scores_abs = scores_abs,
                        edges_indices = edges_indices,
                        edges_costs = edges_costs
                        )
                    )
            outputs.append(result)

        if self.return_belive_maps:
            outputs.append((xhat, PAFs))
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs
    
    #%%
    
    
    
    