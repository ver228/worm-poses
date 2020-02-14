#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:07:18 2020

@author: avelinojaver
"""
from pathlib import Path
import tqdm
import tables
import numpy as np
import pickle
import multiprocessing as mp


def _process_file(fname):
    with tables.File(fname, 'r') as fid:
        tab = fid.get_node('/trajectories_data')
        microns_per_pixel = tab._v_attrs['microns_per_pixel']
        fps = tab._v_attrs['fps']
        
        timeseries_data = fid.get_node('/timeseries_data')
        lengths = [row['length'] for row in timeseries_data]
        widths_t = fid.get_node('/coordinates/widths')[:]
        
    length = np.nanmedian(lengths)
    widths = np.nanmedian(widths_t, axis=0)
    
    
    return fname.name, microns_per_pixel, fps, length, widths

if __name__ == '__main__':
    
    
    root_dir = Path.home() / 'workspace/WormData/screenings/single_worm/finished'
    
    save_name = Path.home() / 'workspace/WormData/screenings/single_worm/median_widths.p'
    
    fnames = list(root_dir.rglob('*_featuresN.hdf5'))
    all_data = []
    
    with mp.Pool(4) as p:
      all_data = list(tqdm.tqdm(p.imap(_process_file, fnames), total = len(fnames)))
    
        
    with open(save_name, 'wb') as fid:
        pickle.dump(all_data, fid)
        