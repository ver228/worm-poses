#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
__root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(__root_dir))

import random
import numpy as np
import tables
import torch
import torch.nn.functional as F
import tqdm
import time
import pandas as pd
import imgstore

import multiprocessing as mp
mp.set_start_method('spawn', force = True)
from worm_poses.utils import get_device
from worm_poses.models import PoseDetector

from process_with_PAF import _process_from_reader, get_model_arguments

def _prepare_batch(batch):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    X = torch.from_numpy(X).unsqueeze(1)
    return frames, X
    
    
def read_images_loopbio(file_name, batch_size, queue):
    
    
    store = imgstore.new_for_filename(str(file_name))
    bn = Path(file_name).parent.name
    
    
    batch = []
    for frame_number in tqdm.trange(store.frame_count, desc = bn):
        img = store.get_next_image()[0]
        batch.append((frame_number, img))
        if len(batch) >= batch_size:
            frames, X = _prepare_batch(batch)
            batch = []
            queue.put((frames, X))
    
    if len(batch) > 0:
        frames, X = _prepare_batch(batch)
        queue.put((frames, X))
    
            
    queue.put(None)
    
    while not queue.empty():
        #wait until the consumer empty the queue before destroying the process.
        time.sleep(1)


def _init_reader(mask_file, batch_size, device, images_queue_size):
    queue_images = mp.Queue(images_queue_size)
    
    reader_p = mp.Process(
                          target = read_images_loopbio, 
                          args= (mask_file, batch_size, queue_images)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    return reader_p, queue_images
    

def _process_file(mask_file, save_name, model, device, batch_size, images_queue_size = 4, resize_factor = None):
    reader_p, queue_images = _init_reader(mask_file, batch_size, device, images_queue_size)
    try:
        _process_from_reader(queue_images, save_name, model, device, resize_factor = resize_factor)
    except Exception as e:
        reader_p.terminate()
        raise e
    


if __name__ == '__main__':
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+head_maxlikelihood_20191219_150412_adam_lr0.0001_wd0.0_batch20'
    #bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200118_100732_adam_lr0.0001_wd0.0_batch24'
    
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    #save_dir_root = Path.home() / 'workspace/WormData/worm-poses/processed'
    #save_dir = save_dir_root / bn
    #save_dir.mkdir(exist_ok = True, parents = True)
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 2#5#3
    
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
    root_dir = Path.home() / 'workspace/WormData/screenings/hydra_example_for_Avelino/' 
    
    videos_dir = root_dir / 'RawVideos'
    save_root_dir = root_dir / f'LoopBio_ResultsNN_{bn}'
    
    fnames = [x for x in videos_dir.rglob('metadata.yaml') if not x.name.startswith('.')]
    #%%
    random.shuffle(fnames)
    
    unprocessed_files = []
    for fname in fnames:
        base_name = fname.parent.name
        
        save_dir = Path(str(fname.parents[1]).replace(str(videos_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        save_name = save_dir / (base_name + '_unlinked-skels.hdf5') 
        
        if save_name.exists():
            continue
        try:
            _process_file(fname, save_name, model, device, batch_size)
        except Exception as e:
            unprocessed_files.append((fname, str(e)))
            
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)
    
        
    