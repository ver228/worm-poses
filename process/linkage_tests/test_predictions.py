#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 00:51:10 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from link_skeletons import _link_points_to_segments

import tables
import matplotlib.pylab as plt
import numpy as np




root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/'
root_dir = Path(root_dir)


#bn = 'CX11271_Ch1_30092017_080938'
bn = 'JU792_Ch1_24092017_063115'
#bn = 'ED3017_Ch1_01072017_102004'
#bn = 'CX11314_Ch2_01072017_093003'
#bn = 'JU2565_Ch2_27082017_112328'


mask_file = root_dir / f'MaskedVideos/{bn}.hdf5'
#pred_files = root_dir / 'Results' / (mask_file.stem + '_unlinked-skels-rois.hdf5')
#pred_files = root_dir / 'Results' / (mask_file.stem + '_unlinked-skels.hdf5')
#pred_files = root_dir / 'ResultsNN_v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22' / (mask_file.stem + '_unlinked-skels.hdf5')
#pred_files = root_dir / 'ResultsNN_v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32' / (mask_file.stem + '_unlinked-skels.hdf5')


mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/egl-43 (n1079)II on food L_2010_08_19__11_57_42__5.hdf5'
pred_files = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/egl-43 (n1079)II on food L_2010_08_19__11_57_42__5_unlinked-skels.hdf5'


#%%
with tables.File(pred_files) as fid:
    points = fid.get_node('/points')[:]
    edges = fid.get_node('/edges')[:]


frame = 16125#2656#11209#18915#
points_in_frame = points[points['frame_number'] == frame]
with tables.File(mask_file) as fid:
    img = fid.get_node('/mask')[frame]

plt.figure()
plt.imshow(img, cmap = 'gray')
#plt.plot(points_in_frame['x'], points_in_frame['y'], '.r')

cmx = np.mean(points_in_frame['x'])
cmy = np.mean(points_in_frame['y'])
plt.xlim((cmx - 100, cmx + 100))
plt.ylim((cmy - 100, cmy + 100))


#%%
# frame = 0
# with tables.File(mask_file) as fid:
#     img = fid.get_node('/mask')[frame]
#%%
#frames = np.unique(points['frame_number'])
# for frame in frames:
#     points_in_frame = points[points['frame_number'] == frame]
#     edgs_in_frame = edges[edges['frame_number'] == frame]
    
#     top_points = points_in_frame[np.argsort(points_in_frame['score'])[-5:]]
    
    
    
#     plt.figure()
#     plt.imshow(img, cmap = 'gray')
#     plt.plot(points_in_frame['x'], points_in_frame['y'], '.r')
    
#     cmx = np.mean(top_points['x'])
#     cmy = np.mean(top_points['y'])
#     plt.plot(cmx, cmy, 'ob')
#%%
import random
n_segments = 15
tot_frames = points['frame_number'].max() + 1
points_g = [[[] for _ in range(n_segments)] for _ in range(tot_frames)]
for irow, row in enumerate(points):
    points_g[row['frame_number']][row['segment_id']].append(irow)

edges_g = [[] for _ in range(tot_frames)]
for irow, row in enumerate(edges):
    edges_g[row['frame_number']].append(irow)
segments_linked = _link_points_to_segments(frame, points, points_g, edges, edges_g, n_segments)

for seg in segments_linked:
    seg = np.array(seg)
    
    delx = (random.random()-0.5)*1e-0
    dely = (random.random()-0.5)*1e-0
    plt.plot(seg['x'] + delx, seg['y'] + dely, '.-')


#%%




#%%
# x,y,w = points_in_frame['x'], points_in_frame['y'], points_in_frame['score']
# w /= np.sum(w)
# plt.plot(np.sum(w*x), np.sum(w*y), 'sc')
# plt.plot(np.mean(x), np.mean(y), 'vy')

