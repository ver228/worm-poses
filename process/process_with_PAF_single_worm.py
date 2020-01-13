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

import multiprocessing as mp
mp.set_start_method('spawn', force = True)

from worm_poses.utils import get_device
from worm_poses.models import PoseDetector

def _prepare_batch(batch, device):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    X = torch.from_numpy(X).unsqueeze(1)
    #X = X.to(device)
    return frames, X
    
    
def read_images_proc(mask_file, batch_size, queue, device):
    
    bn = Path(mask_file).stem
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_h, img_w = masks.shape
        
        for frame_number in tqdm.trange(0, tot, batch_size, desc = bn):
            img = masks[frame_number:frame_number + batch_size]
            img = img.astype(np.float32)/255.
            
            X = torch.from_numpy(img)
            X = X.unsqueeze(1)
            queue.put((frame_number, X))
            
    queue.put(None)
    
    while not queue.empty():
        #wait until the consumer empty the queue before destroying the process.
        time.sleep(1)



        
#%%
def _get_roi_sizes(skel_preds, roi_size, img_lims):
    cm = np.median(skel_preds[:, 1:], axis=0).round().astype(np.int)     
    xl, yl = cm - roi_size//2
    xl = min(max(0, xl), img_lims[1]-roi_size)
    yl = min(max(0, yl), img_lims[0]-roi_size)
    xr = xl + roi_size
    yr = yl + roi_size
    
    return (yl, yr), (xl, xr)
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

def _process_file(mask_file, save_file_roi, save_file_full, model, device, batch_size, roi_size):    
    queue_images = mp.Queue(images_queue_size)
    
    reader_p = mp.Process(target = read_images_proc, 
                          args= (mask_file, batch_size, queue_images, device)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    
    acc_full = predictionsAccumulator()
    acc_roi = predictionsAccumulator()
    
    
    
    with torch.no_grad():
        while True:
            try:
                dat = queue_images.get()
            except ConnectionResetError:
                break
            
            if dat is None:
                break
            frame, batch = dat
            
            X = batch[:1]
            X = X.to(device)
            full_predictions = model(X)
            full_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in full_predictions]
            acc_full.add([frame], full_predictions)
            
            
            img_lims = X.shape[-2:]
            preds = full_predictions[0]['skeletons']
            rlims = _get_roi_sizes(preds, roi_size, img_lims)
            (yl, yr), (xl, xr) = rlims
            
            X_roi = batch[..., yl:yr, xl:xr]
            X_roi = X_roi.to(device)
            roi_predictions = model(X_roi)
            roi_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in roi_predictions]
            
            frames = np.arange(frame, frame + len(roi_predictions))
            acc_roi.add(frames, roi_predictions, corner = (yl, xl))
            
    acc_roi.save(save_file_roi)
    acc_full.save(save_file_full)
if __name__ == '__main__':
    bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 128
    roi_size = 256
    
    images_queue_size = 2
    results_queue_size = 4
    
    model_args = dict(
        n_segments = 8,
        n_affinity_maps = 8,
        features_type = 'vgg11',
        n_stages = 4,
    )
    #%%
    model = PoseDetector(**model_args)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    model = model.to(device)
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/screenings/Bertie_movies'
    
    mask_dir = root_dir / 'MaskedVideos'
    save_root_dir = root_dir / 'ResultsNN'
    
    mask_files = list(mask_dir.rglob('*.hdf5'))
    random.shuffle(mask_files)
    
    unprocessed_files = []
    for mask_file in mask_files:
        bn = mask_file.stem
        save_dir = Path(str(mask_file.parent).replace(str(mask_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        
        save_file_roi = save_dir / (mask_file.stem + '_unlinked-skels-rois.hdf5') 
        save_file_full = save_dir / (mask_file.stem + '_unlinked-skels-fulls.hdf5') 
        
        if save_file_full.exists():
            continue
        try:
            _process_file(mask_file, save_file_roi, save_file_full, model, device, batch_size, roi_size)
        except Exception as e:
            unprocessed_files.append((mask_file, e))
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)