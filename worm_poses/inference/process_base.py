#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""

import numpy as np
import tables
import torch
import torch.nn.functional as F
import pandas as pd

from .readers import read_images_from_tierpsy, init_reader

@torch.no_grad()
def process_from_reader(
                queue_images,
                  save_name, 
                  model, 
                  device,
                  resize_factor = None
                  ):
    
    edges = []
    points = []
    current_poind_id = 0
    while True:
        dat = queue_images.get(timeout = 60)
        if dat is None:
            break
        frames, X = dat
        
        if resize_factor is not None:
            X = F.interpolate(X, scale_factor = resize_factor)
        
        X = X.to(device)
        predictions = model(X)
        predictions = [{k : v.cpu().numpy() for k,v in p.items()} for p in predictions]
        
        for frame, prediction in zip(frames, predictions):
            
            seg_id, skel_x, skel_y = prediction['skeletons'].T
            nms_score = prediction['scores_abs'].T
            edge_p1, edge_p2 = prediction['edges_indices']
            edge_cost_PAF, edge_cost_R = prediction['edges_costs']
            
            
            N = len(seg_id)
            t_p = np.full(N, frame)
            p_ids = range(current_poind_id, current_poind_id + N)
            points += list(zip(p_ids, t_p, seg_id, skel_x, skel_y, nms_score))
            
            t_e = np.full(len(edge_p1), frame)
            edge_p1 += current_poind_id
            edge_p2 += current_poind_id
            edges += list(zip(t_e, edge_p1, edge_p2, edge_cost_PAF, edge_cost_R))
            
            current_poind_id += N
            
    
    points_df = pd.DataFrame(points, columns = ['point_id', 'frame_number', 'segment_id', 'x', 'y', 'score'])
    if resize_factor is not None:
        points_df[['x', 'y']] /= resize_factor
    
    edges_df = pd.DataFrame(edges, columns = ['frame_number', 'point1', 'point2', 'cost_PAF', 'cost_R'])
    
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

def process_file(mask_file, save_name, model, device, batch_size, images_queue_size = 4, resize_factor = None, reader_func=read_images_from_tierpsy):
    reader_p, queue_images = init_reader(reader_func, mask_file, batch_size, device, images_queue_size)
    try:
        process_from_reader(queue_images, save_name, model, device, resize_factor = resize_factor)
    except Exception as e:
        reader_p.terminate()
        raise e