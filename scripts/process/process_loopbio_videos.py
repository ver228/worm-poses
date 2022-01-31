#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""
from pathlib import Path 

import random
import torch

import multiprocessing as mp
mp.set_start_method('spawn', force = True)
from worm_poses.utils import get_device
from worm_poses.inference import load_model, process_file, read_images_loopbio


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
    
    
    model = load_model(model_path, device)
    
    process_args = dict(
        model=model, 
        device=device, 
        batch_size=batch_size, 
        images_queue_size=images_queue_size, 
        reader_func=read_images_loopbio
    )
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
            process_file(fname, save_name, **process_args)
        except Exception as e:
            unprocessed_files.append((fname, str(e)))
            
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)
    
        
    