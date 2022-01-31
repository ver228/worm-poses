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

import numpy as np
import tables
import torch
import tqdm

import multiprocessing as mp

from worm_poses.utils import get_device
from worm_poses.models import PoseDetector

def _prepare_batch(batch, device):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    X = torch.from_numpy(X).unsqueeze(1)
    X = X.to(device)
    
    return frames, X
    
    
def read_images_proc(mask_file, batch_size, queue, device):
    
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_h, img_w = masks.shape
        
        bn = mask_file.stem
        
        batch = []
        for frame_number in tqdm.trange(masks.shape[0], desc = bn):
            img = masks[frame_number]
            if not img.any():
                continue
            
            batch.append((frame_number, img))
            if len(batch) >= batch_size:
                queue.put(_prepare_batch(batch, device))
                batch = []
            
                
    if batch:
        queue.put(_prepare_batch(batch, device))
    queue.put(None)


if __name__ == '__main__':
    
    bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 5
    
    gpu_queue_size = 2
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
    #%%
    test_file = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5'
    
    mp.set_start_method('spawn')
    pqueue = mp.Queue(gpu_queue_size)
    reader_p = mp.Process(target = read_images_proc, 
                          args= (test_file, batch_size, pqueue, device)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process

    while True:
        dat = pqueue.get()
        if dat is None:
            break
        frames, X = dat
        
        with torch.no_grad():
            predictions_o = model(X)
            predictions = [{k : v.cpu().numpy() for k,v in p.items()} for p in predictions_o]
            
        