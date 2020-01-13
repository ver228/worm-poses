#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:31:46 2019

@author: avelinojaver
"""
from pathlib import Path
import torch
import random
from process_with_PAF import _process_file, get_device, PoseDetector
from tqdm import tqdm

if __name__ == '__main__':
    bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    
    model_name = bn.split('_')[1]
    if '+light' in model_name:
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
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    #save_dir_root = Path.home() / 'workspace/WormData/worm-poses/processed'
    #save_dir = save_dir_root / bn
    #save_dir.mkdir(exist_ok = True, parents = True)
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 48#36#16#
    
     
    
    #%%
    model = PoseDetector(**model_args)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    #root_dir = Path.home() / 'workspace/WormData/screenings/mating_videos/wildMating/'
    #root_dir = Path.home() / 'workspace/WormData/screenings/mating_videos/Mating_Assay/'
    
    root_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
    mask_dir = root_dir / 'finished'
    save_root_dir = root_dir / 'ResultsNN'
    
    mask_files = list(mask_dir.rglob('*.hdf5'))
    
    _bad_ext = ['interpolated25', 'featuresN']
    mask_files = [x for x in mask_files if not (x.stem.rpartition('_')[-1] in _bad_ext)]
    
    random.shuffle(mask_files)
    
    unprocessed_files = []
    
    tot = 0
    for mask_file in tqdm(mask_files):
        bn = mask_file.stem
        save_dir = Path(str(mask_file.parent).replace(str(mask_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        save_name = save_dir / (mask_file.stem + '_unlinked-skels.hdf5') 
        
        if save_name.exists():
            tot += 1
            continue
        
        _process_file(mask_file, save_name, model, device, batch_size)
        # try:
        #     _process_file(mask_file, save_name, model, device, batch_size)
        # except Exception as e:
        #     unprocessed_files.append((mask_file, e))
    print(f'Exists {tot}')
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)