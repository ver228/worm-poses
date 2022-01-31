#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""
from pathlib import Path
import random

from worm_poses.utils import get_device
from worm_poses.inference import load_model, process_file, read_images_from_tierpsy, unlinked2linked_skeletons

if __name__ == '__main__':
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
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
        reader_func=read_images_from_tierpsy
    )

    #%%
    #root_dir = Path.home() / 'workspace/WormData/screenings/mating_videos/wildMating/'
    root_dir = Path.home() / 'workspace/WormData/screenings/mating_videos/Mating_Assay/'
    #root_dir = Path.home() / 'workspace/WormData/screenings/Pratheeban/'
    #root_dir = Path.home() / 'workspace/WormData/screenings/Serena_WT_Screening/'
    #root_dir = Path.home() / 'workspace/WormData/screenings/pesticides_adam/Syngenta/'
    
    #root_dir = Path.home() / 'workspace/WormData/screenings/hydra_example_for_Avelino/' 
    
    mask_dir = root_dir / 'MaskedVideos'
    save_root_dir = root_dir / f'ResultsNN_{bn}'
    
    mask_files = [x for x in mask_dir.rglob('*.hdf5') if not x.name.startswith('.')]
    
    #files2include = ['wildMating3.2_MY23_cross_MY23_cross_PC1_Ch1_17082018_123407.hdf5', 'Set1_CB369_CB1490_Ch2_22062018_120002.hdf5', 'wildMating3.1_MY23_self_MY23_self_PC1_Ch1_17082018_121653.hdf5', 'Set2_N2_PS3398_Ch2_03072018_170416.hdf5', 'wildMating1.2_CB4856_cross_CB4856_cross_PC3_Ch2_15082018_121405.hdf5', 'wildMating3.1_CB4856_self_CB4856_self_PC3_Ch2_17082018_114619.hdf5', 'CB369_PS3398_Set2_Pos5_Ch2_180608_144551.hdf5', 'wildMating4.2_MY23_self_CB4856_self_PC2_Ch2_18082018_125140.hdf5']
    #files2include = ['15.1_5_cx11314_da_cb4852_ff_Set0_Pos0_Ch1_03032018_145825.hdf5', '15.1_2_n2_6b_Set0_Pos0_Ch2_03032018_113804.hdf5']
    #files2include = ['N2_worms10_CSAA026102_100_Set7_Pos4_Ch3_14072017_195800.hdf5']
    #mask_files = [x for x in mask_files if x.name in files2include]
    
    
    random.shuffle(mask_files)
    
    unprocessed_files = []
    for mask_file in mask_files:
        bn = mask_file.stem
        save_dir = Path(str(mask_file.parent).replace(str(mask_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        save_name = save_dir / (mask_file.stem + '_unlinked-skels.hdf5') 
        
        if save_name.exists():
            continue
        #try:
        process_file(mask_file, save_name, **process_args)
        unlinked2linked_skeletons(save_name)
        #except Exception as e:
        #    unprocessed_files.append((mask_file, str(e)))
            
    #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)
    
        
    