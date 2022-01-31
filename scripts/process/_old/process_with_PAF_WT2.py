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
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    #save_dir_root = Path.home() / 'workspace/WormData/worm-poses/processed'
    #save_dir = save_dir_root / bn
    #save_dir.mkdir(exist_ok = True, parents = True)
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    #36#16#
    
    #%%
    resize_factor = 0.5
    batch_size = 96
    model = PoseDetector(**model_args, keypoint_max_dist = 20, nms_min_distance = 5)
    #model = PoseDetector(**model_args, keypoint_max_dist = 100, nms_min_distance = 5)
    #batch_size = 48
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/screenings/single_worm/'
    mask_dir = root_dir / 'finished'
    save_root_dir = root_dir / ('ResultsNN_' + bn)
    
    mask_files = list(mask_dir.rglob('*.hdf5'))
    
    _bad_ext = ['interpolated25', 'featuresN']
    mask_files = [x for x in mask_files if not (x.stem.rpartition('_')[-1] in _bad_ext)]
    
    files2include = ['197 PS312 4 on N2-LL2 R_2011_06_17__15_28___3___3.hdf5',
                      'unc-76 (e911)V on food R_2011_09_22__11_34_33___8___2.hdf5',
                      'QT309 (nRH01) on food L_2011_10_04__12_15___3___1.hdf5',
                      'unc-34 (e566)V on food L_2010_09_24__12_37_12___6___5.hdf5',
                      'unc-17 (e245)IV on food R_2010_04_16__11_18_53___2___3.hdf5',
                      'egl-43 (n1079)II on food L_2010_08_19__11_57_42__5.hdf5',
                      'nlp-2 (tm1908)X on food R_2010_03_12__12_45_35___8___9.hdf5',
                      'N2 male on food L_2012_02_23__12_05_02___5___2.hdf5',
                      'unc-104 (e1265)III on food L_2011_10_18__13_23_55___1___7.hdf5',
                      'unc-4 (e120)II on food L_2011_10_18__12_47_06___2___5.hdf5']
    
    #files2include = ['N2 male on food L_2012_02_23__12_05_02___5___2.hdf5']
    
    mask_files = [x for x in mask_files if x.name in files2include]
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
        
        #_process_file(mask_file, save_name, model, device, batch_size)
        try:
            _process_file(mask_file, save_name, model, device, batch_size, resize_factor = resize_factor)
        except Exception as e:
            unprocessed_files.append((mask_file, e))
    print(f'Exists {tot}')
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)