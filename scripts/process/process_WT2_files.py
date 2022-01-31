#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:07:03 2019

@author: avelinojaver
"""

    
if __name__ == '__main__':
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+head_maxlikelihood_20191223_105436_adam_lr0.0001_wd0.0_batch28'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200118_100732_adam_lr0.0001_wd0.0_batch24'
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    batch_size = 64
    roi_size = 256
    
    images_queue_size = 2
    results_queue_size = 4
    
    model = load_model(model_path, device)
    
    #%%
    root_dir = Path.home() / 'workspace/WormData/screenings/Bertie_movies'
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies')
    
    mask_dir = root_dir / 'MaskedVideos'
    save_root_dir = root_dir / ('ResultsNN_' + bn)
    
    #mask_files = list(mask_dir.rglob('JU2565_Ch2_27082017_112328*.hdf5'))
    mask_files = list(mask_dir.rglob('*.hdf5'))
    random.shuffle(mask_files)
    
    unprocessed_files = []
    for mask_file in mask_files:
        bn = mask_file.stem
        save_dir = Path(str(mask_file.parent).replace(str(mask_dir), str(save_root_dir)))
        save_dir.mkdir(exist_ok = True, parents = True)
        save_name = save_dir / (mask_file.stem + '_unlinked-skels.hdf5') 
        
        #_process_file(mask_file, save_name, model, device, batch_size)
        if save_name.exists():
            continue
        try:
            _process_file(mask_file, save_name, model, device, batch_size)
            with tables.File(save_name, 'r+') as fid:
                fid.get_node('/points')._v_attrs['src_model'] = str(bn)
            
        except Exception as e:
            unprocessed_files.append((mask_file, str(e)))
    
    
        #I couldn't process the following files
    for fname, e in unprocessed_files:
        print(fname)
        print(e)