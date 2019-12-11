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
from worm_poses.models.detection_torch import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn

def _prepare_batch(batch):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    X = np.repeat(X[:, None], 3, axis=1)
    X = torch.from_numpy(X)
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
    
    #bn = 'detection-singles_fasterrcnn_20190628_174616_adam_lr0.0001_wd0.0_batch16'
    bn = 'detection-singles_keypointrcnn_20190703_085950_adam_lr0.0001_wd0.0_batch8'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'checkpoint.pth.tar'
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 6
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    
    
    roi_size = 2048
    if 'keypointrcnn' in bn:
        model = keypointrcnn_resnet50_fpn(
                                        num_classes = 2, 
                                        num_keypoints = 25,
                                        min_size = roi_size,
                                        max_size = roi_size,
                                        image_mean = [0., 0., 0.],
                                        image_std = [1., 1., 1.], 
                                        pretrained = False
                                        )
    elif 'fasterrcnn' in bn:
        model = fasterrcnn_resnet50_fpn(
                                    num_classes = 2, 
                                    min_size = roi_size,
                                    max_size = roi_size,
                                    image_mean = [0, 0, 0],
                                    image_std = [1., 1., 1.],
                                    pretrained = False
                                    )
    
    model.load_state_dict(state['state_dict'])
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
                
                            