#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
_script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(_script_dir))

import random
import numpy as np
import tables
import torch
import pandas as pd
from collections import defaultdict

import multiprocessing as mp
mp.set_start_method('spawn', force = True)

from worm_poses.utils import get_device
from worm_poses.models import PoseDetector

from process_with_PAF import _init_reader, get_model_arguments


        
#%%
def _get_roi_limits(skel_preds, roi_size, img_lims, corner = None):
    cm = [np.median(p[:, -2:], axis=0).round().astype(np.int) for p in skel_preds]
    cm = np.mean(cm, axis=0)  
    
    xl, yl = cm - roi_size//2
    
    if corner is not None:
        xl += corner[0]
        yl += corner[1]
    
    
    xl = int(min(max(0, xl), img_lims[1]-roi_size))
    yl = int(min(max(0, yl), img_lims[0]-roi_size))
    xr = xl + roi_size
    yr = yl + roi_size
    
    
    return (yl, yr), (xl, xr)

#%%
def predictions2skeletons(preds, 
                          n_segments = 8,
                          min_PAF = 0.25,
                          is_skel_half = True,
                          _frame2test = -1
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
    
    #add the point_id column 
    assert (edges_indeces < len(points)).all()
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
    
    # if _frame2test == 143:
    #     import pdb
    #     pdb.set_trace()
        
    
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
                skel_x = skel_x + corner[0]
                skel_y = skel_y + corner[1]
                
            N = len(seg_id)
            t_p = np.full(N, frame)
            p_ids = range(self.current_poind_id, self.current_poind_id + N)
            self.points += list(zip(p_ids, t_p, seg_id, skel_x, skel_y, nms_score))
            
            t_e = np.full(len(edge_p1), frame)
            edge_p1 = edge_p1 + self.current_poind_id
            edge_p2 = edge_p2 + self.current_poind_id
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
def _process_from_reader_single(queue_images, save_name, model, device, full_batch_size = 2 ):
    
    pred_acc = predictionsAccumulator()
    
    def skelfuncs(x, _frame2test):
        return predictions2skeletons(x, 
                                    _frame2test = _frame2test,
                                     n_segments = model.n_segments,
                                     is_skel_half = True
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
                frames_in_batch = frames[:full_batch_size]
                
                batch = batch[full_batch_size:]
                frames = frames[full_batch_size:]
                
                X = X.to(device)
                full_predictions = model(X)
                full_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in full_predictions]
                
                pred_acc.add(frames_in_batch, full_predictions)
                
                
                
                skels = [skelfuncs(p, _frame2test = frame) for p, frame in zip(full_predictions, frames_in_batch)]
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
            
            _corner = (xl, yl)
            pred_acc.add(frames, roi_predictions, corner = _corner)
            preds = [x['skeletons'] for x in roi_predictions[-10:]]
            roi_limits = _get_roi_limits(preds, roi_size, img_lims, corner = _corner)
    
           
    pred_acc.save(save_name)



def _process_file(mask_file, save_name, model, device, batch_size, images_queue_size = 4):
    reader_p, queue_images = _init_reader(mask_file, batch_size, device, images_queue_size)
    try:
        _process_from_reader_single(queue_images, save_name, model, device)
    except Exception as e:
        reader_p.terminate()
        raise e

    
if __name__ == '__main__':
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+head_maxlikelihood_20191223_105436_adam_lr0.0001_wd0.0_batch28'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200118_100732_adam_lr0.0001_wd0.0_batch24'
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 64
    roi_size = 256
    
    images_queue_size = 2
    results_queue_size = 4
    
    model_args = get_model_arguments(bn)
    #%%
    model = PoseDetector(**model_args)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/screenings/Bertie_movies'
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies')
    
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
        save_name = save_dir / (mask_file.stem + '_unlinked-skels.hdf5') 
        
        #_process_file(mask_file, save_name, model, device, batch_size)
        if save_name.exists():
            continue
        try:
            _process_file(mask_file, save_name, model, device, batch_size)
            with tables.File(save_name, 'r+') as fid:
                fid.get_node('/points')._v_attrs['src_model'] = str(bn)
            
        except Exception as e:
            unprocessed_files.append((mask_file, str(e)))
    
    
        #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)