#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:59:26 2019

@author: avelinojaver
"""
import random
import pandas as pd
import math
from pathlib import Path

if __name__ == '__main__':
    base_dir = '/users/rittscher/avelino/workspace/WormData/worm-poses/labelled_rois/'
    
    files_list_f = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/labelled_rois/all_files.txt'
    
    with open(files_list_f, 'r') as fid:
        fnames = fid.read().split('\n')
    
    fnames = [x.replace(base_dir, '') for x in fnames if x and not x.rpartition('/')[-1].startswith('.')]
    #%%
    files_data = {}
    for fname in fnames:
        set_k = fname.partition('/')[0]
        movie_id = fname.rpartition('/')[-1].partition('_')[0]
        
        if set_k not in files_data:
            files_data[set_k] = {}
        
        if movie_id not in files_data[set_k]:
            files_data[set_k][movie_id] = []
        
        files_data[set_k][movie_id].append(fname)
    #%%
    test_frac = 0.05
    
    train_set = []
    test_set = []
    for set_k, set_data in files_data.items():
        movie_ids = list(set_data.keys())
        random.shuffle(movie_ids)
        cut_ind = math.ceil(len(movie_ids)*test_frac)
        
        for m_ii, movie_id in enumerate(movie_ids):
            f_list = set_data[movie_id]
            dd = [(set_k, movie_id, x) for x in f_list]
            
            if m_ii <= cut_ind:
                test_set += dd
            else:
                train_set += dd
        
        
    #%%
    columns = ['label_set', 'movie_id', 'file_path']
    train_df = pd.DataFrame(train_set, columns = columns)
    test_df = pd.DataFrame(test_set, columns = columns)
    
    
    save_dir = Path.home() / 'workspace/WormData/skeletonize_training/labelled_rois/'
    
    train_df.to_csv(str(save_dir / 'train_list.csv'), index = False)
    test_df.to_csv(str(save_dir / 'test_list.csv'), index = False)