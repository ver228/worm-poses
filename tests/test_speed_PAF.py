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

def _prepare_batch(batch):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    
    X = torch.from_numpy(X[:, None])
    return frames, X
    
    
def read_images_proc(mask_file, batch_size, queue):
    
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot, img_h, img_w = masks.shape
        
        bn = mask_file.stem
        
        batch = []
        for frame_number in tqdm.trange(masks.shape[0], desc = bn):
            img = masks[frame_number]
            
            batch.append((frame_number, img))
            if len(batch) >= batch_size:
                queue.put(_prepare_batch(batch))
                batch = []
            
                
    if batch:
        queue.put(_prepare_batch(batch))
    queue.put(None)


if __name__ == '__main__':
    
    bn = 'v2_openpose_maxlikelihood_20191209_093737_adam_lr0.0001_wd0.0_batch20'
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'checkpoint.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 2
    
    #%%
    
    n_stages = 4
    n_segments = 17
    PAF_seg_dist = 2
    features_type = 'vgg11'
    #%%
    #n_stages = 6
    #n_segments = 49
    #PAF_seg_dist = 5
    #features_type = 'vgg19'
    
    n_affinity_maps = 2*(n_segments//2 - PAF_seg_dist + 1) + 1
    #%%
    model = PoseDetector(
            n_stages = n_stages, 
            n_segments = n_segments,
            n_affinity_maps = n_affinity_maps, 
            features_type = features_type,
            return_belive_maps = True,
            same_output_size = False
            )
    
    #state = torch.load(model_path, map_location = 'cpu')
    #model.load_state_dict(state['state_dict'])
    
    model.eval()
    model = model.to(device)
    #%%
    test_file = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5'
    
    pqueue = mp.Queue(batch_size)
    reader_p = mp.Process(target = read_images_proc, 
                          args= (test_file, batch_size, pqueue)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process

    while True:
        dat = pqueue.get()
        if dat is None:
            break
        frames, X = dat
        
        with torch.no_grad():
            
            X = X.to(device)
            
            predictions = model(X)
            #preditions = [{k:v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
                
    # with torch.no_grad():
    #     for _ in tqdm.trange(1000):
    #         X = torch.rand((batch_size, 1, 2048, 2048))
    #         X = X.to(device)
    #         predictions = model(X)