#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import random
import numpy as np
import tables
import torch
import tqdm
import time
import pandas as pd
from collections import defaultdict

import multiprocessing as mp
mp.set_start_method('spawn', force = True)

from worm_poses.utils import get_device
from worm_poses.models import PoseDetector


# def _prepare_batch(batch, device):
#     frames, X = map(np.array, zip(*batch))
#     X = X.astype(np.float32)/255.
#     X = torch.from_numpy(X).unsqueeze(1)
#     #X = X.to(device)
#     return frames, X
    
    
# def read_images_proc(mask_file, batch_size, queue, device):
#     bn = Path(mask_file).stem
#     with tables.File(mask_file, 'r') as fid:
#         masks = fid.get_node('/mask')
#         tot, img_h, img_w = masks.shape
        
#         batch = []
#         for frame_number in tqdm.trange(tot, desc = bn):
#             img = masks[frame_number]
#             if not img.any():
#                 continue
            
#             batch.append((frame_number, img))
#             if len(batch) >= batch_size:
#                 queue.put(_prepare_batch(batch, device))
#                 batch = []
                 
#     if len(batch):
#         queue.put(_prepare_batch(batch, device))
#     queue.put(None)
    
#     while not queue.empty():
#         #wait until the consumer empty the queue before destroying the process.
#         time.sleep(1)

def read_images_batch_proc(mask_file, batch_size, queue, device):
    bn = Path(mask_file).stem
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_h, img_w = masks.shape
        
        for frame_number in tqdm.trange(0, tot, batch_size, desc = bn):
            X = masks[frame_number:frame_number + batch_size]
            X = X.astype(np.float32)/255.
            X = torch.from_numpy(X).unsqueeze(1)
            frames = list(range(frame_number, frame_number + X.shape[0]))
            queue.put((frames, X))
            
    queue.put(None)
    
    while not queue.empty():
        #wait until the consumer empty the queue before destroying the process.
        time.sleep(1)

        
#%%
def _get_roi_limits(skel_preds, roi_size, img_lims):
    cm = [np.median(p[:, -2:], axis=0).round().astype(np.int) for p in skel_preds]
    cm = np.mean(cm, axis=0)  
    
    xl, yl = cm - roi_size//2
    xl = int(min(max(0, xl), img_lims[1]-roi_size))
    yl = int(min(max(0, yl), img_lims[0]-roi_size))
    xr = xl + roi_size
    yr = yl + roi_size
    
    return (yl, yr), (xl, xr)

#%%
def predictions2skeletons(preds, 
                          n_segments = 8,
                          min_PAF = 0.2,
                          is_skel_half = True
                          ):
    
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
    
    segments_linked = []
    
    #add the point id 
    points =  np.concatenate((np.arange(len(points))[:, None], points), axis=1)     
    for ipoint in range(n_segments):
        points_in_segment = points[points[:, 1] == ipoint]
        
        prev_inds = {x[-1][0]:x for x in segments_linked}
        
        
        matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
        matched_indices_cur = [best_matches[x] for x in matched_indices_prev]

        new_points = set(points_in_segment[:, 0]) - set(matched_indices_cur)

        try:
            for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
                prev_inds[k1].append(points[k2])
        except:
            import pdb
            pdb.set_trace()
        segments_linked += [[points[x]] for x in new_points]
    
    
    if is_skel_half:
        skeletons = []
        matched_halves = defaultdict(list)
        for _half in segments_linked:
            if len(_half) == n_segments:
                midbody_ind = _half[-1][0]
                matched_halves[midbody_ind].append(_half)
        
        for k, _halves in matched_halves.items():
            if len(_halves) == 2:
                skel = np.array(_halves[0] + _halves[1][-2::-1])
                skel = skel[:, -2:]
                
                skeletons.append(skel)
    else:
        skeletons = [np.array(x)[:, -2:] for x in segments_linked]   
    
        
    return skeletons
#%%
    
class predictionsAccumulator():
    def __init__(self):
        self.edges = []
        self.points = []
        self.current_poind_id = 0
        
    def add(self, frames, predictions, corner = None):
        for frame, preds in zip(frames, predictions):
            #predictions = {k : v.cpu().numpy() for k,v in predictions.items()}
            
            seg_id, skel_x, skel_y = preds['skeletons'].T
            nms_score = preds['scores_abs'].T
            edge_p1, edge_p2 = preds['edges_indices']
            edge_cost_PAF, edge_cost_R = preds['edges_costs']
            
            if corner is not None:
                skel_x += corner[1]
                skel_y += corner[0]
                
            
            N = len(seg_id)
            t_p = np.full(N, frame)
            p_ids = range(self.current_poind_id, self.current_poind_id + N)
            self.points += list(zip(p_ids, t_p, seg_id, skel_x, skel_y, nms_score))
            
            t_e = np.full(len(edge_p1), frame)
            edge_p1 += self.current_poind_id
            edge_p2 += self.current_poind_id
            self.edges += list(zip(t_e, edge_p1, edge_p2, edge_cost_PAF, edge_cost_R))
            
            self.current_poind_id += N
    
    def save(self, save_name):
        points_df = pd.DataFrame(self.points, columns = ['point_id', 'frame_number', 'segment_id', 'x', 'y', 'score'])
        edges_df = pd.DataFrame(self.edges, columns = ['frame_number', 'point1', 'point2', 'cost_PAF', 'cost_R'])
        
        points_rec = points_df.to_records(index = False)
        edges_rec = edges_df.to_records(index = False)
        
        
        TABLE_FILTERS = tables.Filters(
                            complevel=5,
                            complib='zlib',
                            shuffle=True,
                            fletcher32=True)
        
        with tables.File(save_name, 'w') as fid:
            fid.create_table('/', 'points', obj = points_rec, filters = TABLE_FILTERS)
            fid.create_table('/', 'edges', obj = edges_rec, filters = TABLE_FILTERS)

#%%
def _process_file(mask_file, save_file, model, is_skel_half, device, batch_size, roi_size):    
    queue_images = mp.Queue(images_queue_size)
    
    reader_p = mp.Process(target = read_images_batch_proc, 
                          args= (mask_file, batch_size, queue_images, device)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    
    pred_acc = predictionsAccumulator()
    
    
    full_batch_size = 4
    
    
    def skelfuncs(x):
        return predictions2skeletons(x, 
                                     n_segments = model.n_segments,
                                     is_skel_half = is_skel_half
                                     )
    
    roi_limits = None
    with torch.no_grad():
        while True:
            try:
                dat = queue_images.get(timeout = 600)
            except (ConnectionResetError, FileNotFoundError):
                break
            
            if dat is None:
                 break
             
            frames, batch = dat
            
            
            img_lims = batch.shape[-2:]
            
            while (roi_limits is None) and (len(batch) > 0):
            
                X = batch[:full_batch_size]
                X = X.to(device)
                full_predictions = model(X)
                full_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in full_predictions]
                
                pred_acc.add(frames, full_predictions)
                
                batch = batch[full_batch_size:]
                frames = frames[full_batch_size:]
                
                skels = [skelfuncs(p) for p in full_predictions]
                skels = [s[0] for s in skels if len(s) == 1]
                
                if skels:
                    roi_limits = _get_roi_limits(skels, roi_size, img_lims)
                    break
            
            if len(batch) == 0:
                continue
            
            (yl, yr), (xl, xr) = roi_limits
            X_roi = batch[..., yl:yr, xl:xr]
            X_roi = X_roi.to(device)
            roi_predictions = model(X_roi)
            roi_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in roi_predictions]
            
            pred_acc.add(frames, roi_predictions, corner = (yl, xl))
            preds = [x['skeletons'] for x in roi_predictions[-10:]]
            roi_limits = _get_roi_limits(preds, roi_size, img_lims)
            
            
    pred_acc.save(save_file)

def get_model_arguments(basename):
    model_name = basename.split('_')[1]
    
    if 'openpose+light+fullsym' == model_name:
        model_args = dict(
            n_segments = 15,
            n_affinity_maps = 14,
            features_type = 'vgg11',
            n_stages = 4,
            nms_min_distance = 1
            )
        
    elif '+light' in model_name:
        model_args = dict(
            n_segments = 8,
            n_affinity_maps = 8,
            features_type = 'vgg11',
            n_stages = 4,
        )
        
        
        
    else:
        model_args = dict(
            n_segments = 8,
            n_affinity_maps = 8,
            features_type = 'vgg19',
            n_stages = 6,
        )
        
    
    model_args['use_head_loss'] = '+head' in model_name
    
    
    is_skel_half = not '+fullsym' in model_name
    
    return model_args, is_skel_half
    
if __name__ == '__main__':
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+head_maxlikelihood_20191223_105436_adam_lr0.0001_wd0.0_batch28'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 96#128
    roi_size = 256
    
    images_queue_size = 2
    results_queue_size = 4
    
    model_args, is_skel_half = get_model_arguments(bn)
    #%%
    model = PoseDetector(**model_args)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/screenings/Bertie_movies'
    
    mask_dir = root_dir / 'MaskedVideos'
    save_root_dir = root_dir / ('ResultsNN_' + bn)
    
    #mask_files = list(mask_dir.rglob('JU2565_Ch2_27082017_112328*.hdf5'))
    mask_files = list(mask_dir.rglob('*.hdf5'))
    random.shuffle(mask_files)
    
    unprocessed_files = []
    for mask_file in mask_files:
        bn = mask_file.stem
        save_dir = Path(str(mask_file.parent).replace(str(mask_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        
        save_file = save_dir / (mask_file.stem + '_unlinked-skels.hdf5') 
        if save_file.exists():
            continue
        _process_file(mask_file, save_file, model, is_skel_half, device, batch_size, roi_size)
        with tables.File(save_file, 'r+') as fid:
            fid.get_node('/points')._v_attrs['src_model'] = bn
        
        
        #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)