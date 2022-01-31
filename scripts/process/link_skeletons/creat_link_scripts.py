#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:53:44 2020

@author: avelinojaver
"""

from pathlib import Path


if __name__ == '__main__':
    save_name = Path.home() / 'workspace/files2link.txt'
    
    root_dir = Path.home() / 'workspace/WormData/screenings'
    results_dir_name = 'ResultsNN_v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    
    src_postfix = '_unlinked-skels.hdf5'
    target_postfix = '_skeletonsNN.hdf5'
    
    subdirs2check  = ['Bertie_movies', 
                      'Serena_WT_Screening', 
                      'Pratheeban', 
                      'mating_videos/Mating_Assay',
                      'mating_videos/wildMating',
                      'pesticides_adam/Syngenta',
                      'single_worm/ResultsNN'
                      ]
    
    files2process = []
    for subdir in subdirs2check:
        dname = root_dir / subdir / results_dir_name
        if not dname.exists():
            dname = root_dir / subdir
        assert dname.exists()
        
        for fname in dname.rglob('*' + src_postfix):
            target_file = fname.parent / (fname.name[:-len(src_postfix)] + target_postfix)
            if not target_file.exists():
                files2process.append(fname)
    #%%
    txt2save = []
    for fname in files2process:
        src_dir = str(fname.parent)
        src_dir = src_dir.replace(str(Path.home()), '')
        prefix = src_dir + '/' + fname.name
        
        txt2save.append(prefix)
                
    with open(save_name, 'w') as fid:
        fid.write('\n'.join(txt2save))
        
        
        
        
    
    