#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:08:17 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from link_skeletons import _calculate_headtail_angles

import tables
import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np

root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/'
root_dir = Path(root_dir)

bn = 'Haw_Ch1_24062017_114319'
#bn = 'JU792_Ch1_24092017_063115'
#bn = 'JU2565_Ch2_27082017_112328'
#bn = 'ED3017_Ch1_01072017_102004'
#bn = 'CX11314_Ch2_01072017_093003'

mask_file = root_dir / f'MaskedVideos/{bn}.hdf5'
#pred_files = root_dir / 'Results' / (mask_file.stem + '_unlinked-skels-rois.hdf5')
pred_files = root_dir / 'Results' / (mask_file.stem + '_unlinked-skels.hdf5')
skel_files = root_dir / 'Results' / (mask_file.stem + '_skeletonsNN.hdf5')

with tables.File(pred_files) as fid:
    points = fid.get_node('/points')[:]
    edges = fid.get_node('/edges')[:]


with pd.HDFStore(skel_files, 'r') as fid:
    trajectories_data = fid['/trajectories_data']
    skeletons = fid.get_node('/coordinates/skeletons')[:]
#%%
skeletons_rev = skeletons[:, ::-1]

skel_disp = np.nanmedian(np.linalg.norm(np.diff(skeletons, axis=0), axis=2), axis=1)
skel_disp_rev = np.nanmedian(np.linalg.norm(skeletons_rev[1:] - skeletons[:-1], axis=2), axis=1)

plt.figure()
plt.plot(skel_disp)
plt.plot(skel_disp_rev)

#%%


inds = np.argsort(skel_disp)[-5:]

inds = [3333]
for ii in inds:
    plt.figure()
    for iskel in range(2):
        skel = skeletons[ii + iskel]
        plt.plot(skel[:, 0], skel[:, 1], 'o-')
        #plt.plot(skel[0, 0], skel[0, 1], 'v')
        
    
    plt.title(f'{ii} - {skel_disp[ii]}')
    plt.axis('equal')
    
#%%
    # skel1 = skeletons[ii]
    # skel2 = skeletons[ii + 1]
    # plt.figure()
    # plt.plot(skel1[:, 0], '.-')
    # plt.plot(skel2[:, 0], '.-')
        
#%%
# frame = 11208
# plt.figure()
# skel = skeletons[frame]
# plt.plot(skel[:, 0], skel[:, 1], 'o-')
# plt.plot(skel[0, 0], skel[0, 1], 'v')

#%%
import math
fps = 25
skel_size = skeletons[0].shape[1]
window_std = max(int(round(fps)), 5)
segment4angle = max(1, int(math.ceil(skel_size / 10)))
angles_head, angles_tail = _calculate_headtail_angles(skeletons, segment4angle)

#%%
plt.figure()
plt.plot(angles_head)
plt.plot(angles_tail)



