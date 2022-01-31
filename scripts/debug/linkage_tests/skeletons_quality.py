#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:20:01 2020

@author: avelinojaver
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


if __name__ == '__main__':
    fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/egl-43 (n1079)II on food L_2010_08_19__11_57_42__5_skeletonsNN.hdf5'
    
    
    with pd.HDFStore(fname, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        skeletons = fid.get_node('/coordinates/skeletons')[:]
    
#%%
    max_gap_size = 12
    for worm_inds, worm_data in trajectories_data.groupby('worm_index_joined'):
        skel_ids = worm_data['skeleton_id'].astype('int').values
        frames = worm_data['timestamp_raw'].astype('int').values
        
        skels = skeletons[skel_ids]
        
        
        print(np.unique(np.isnan(skels).sum(axis=(1,2))))
    
        
    
    #plt.plot(frames, skeletons[:, 0, 0], '.')