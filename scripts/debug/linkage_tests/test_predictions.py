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


root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/Bertie_movies/'

root_dir = Path(root_dir)

#bn = 'ECA396_Ch1_19092017_074741'
#bn = 'CX11254_Ch1_05092017_075253'
#bn = 'CX11271_Ch1_30092017_080938'
#bn = 'JU792_Ch1_24092017_063115'
#bn = 'ED3017_Ch1_01072017_102004'
#bn = 'CX11314_Ch2_01072017_093003'
#bn = 'CX11314_Ch1_04072017_103259'
#bn = 'JU2565_Ch2_27082017_112328'
bn = 'Haw_Ch1_24062017_114319'


mask_file = root_dir / f'MaskedVideos/{bn}.hdf5'
pred_files = root_dir / 'Results' / (mask_file.stem + '_unlinked-skels.hdf5')
#pred_files = root_dir / 'Results_bkp' / (mask_file.stem + '_unlinked-skels.hdf5')

#pred_files = root_dir / 'ResultsNN_v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22' / (mask_file.stem + '_unlinked-skels.hdf5')
#pred_files = root_dir / 'ResultsNN_v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32' / (mask_file.stem + '_unlinked-skels.hdf5')


#mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/egl-43 (n1079)II on food L_2010_08_19__11_57_42__5.hdf5'
#pred_files = '/Users/avelinojaver/OneDrive - Nexus365/worms/WT2/egl-43 (n1079)II on food L_2010_08_19__11_57_42__5_unlinked-skels.hdf5'

# root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/movies/mating/')
# bn = 'wildMating4.2_MY23_self_CB4856_self_PC2_Ch2_18082018_125140'
# bn = 'CB369_PS3398_Set2_Pos5_Ch2_180608_144551'
# mask_file = root_dir / f'{bn}.hdf5'
# pred_files = root_dir / (mask_file.stem + '_unlinked-skels.hdf5')

#mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/WT2/nlp-2 (tm1908)X on food R_2010_03_12__12_45_35___8___9.hdf5'
#pred_files = '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/WT2/nlp-2 (tm1908)X on food R_2010_03_12__12_45_35___8___9_unlinked-skels.hdf5'

mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/hydra_example_for_Avelino/MaskedVideos/syngenta_screen_run1_bluelight_20191214_151529.22956809/metadata.hdf5'
pred_files = '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/hydra_example_for_Avelino/LoopBio_ResultsNN_v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24/syngenta_screen_run1_bluelight_20191214_151529.22956809_unlinked-skels.hdf5'


#%%
with tables.File(pred_files) as fid:
    points = fid.get_node('/points')[:]
    edges = fid.get_node('/edges')[:]




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

#%%


frame = 0#8316#17344#17345#12575#12558#12574#5295#2656#11209#18915#
segments_linked = _link_points_to_segments(frame, points, points_g, edges, edges_g, n_segments)


points_in_frame = points[points['frame_number'] == frame]
with tables.File(mask_file) as fid:
    img = fid.get_node('/mask')[frame]

plt.figure()
plt.imshow(img, cmap = 'gray')
#plt.plot(points_in_frame['x'], points_in_frame['y'], '.r')

cmx = np.mean(points_in_frame['x'])
cmy = np.mean(points_in_frame['y'])
#plt.xlim((cmx - 100, cmx + 100))
#plt.ylim((cmy - 100, cmy + 100))

for seg in segments_linked:
    seg = np.array(seg)
    
    delx = (random.random()-0.5)*1e-0
    dely = (random.random()-0.5)*1e-0
    plt.plot(seg['x'] + delx, seg['y'] + dely, '.-')


#%%

# def _get_best_edge_match(_edges, min_PAF = 0.25):
#     _edges = _edges[_edges['cost_PAF'] >= min_PAF]
#     _edges = _edges[np.argsort(_edges['cost_PAF'])[::-1]] #sort with the largest PAF first...
#     _, valid_index =  np.unique(_edges['point1'], return_index = True ) #unique returns the first occurence (the largest value on the sorted list)
    
#     best_matches = {x['point1']:x['point2'] for x in _edges[valid_index]}
#     return best_matches

# def _link_points_to_segments(_frame, _points, _points_g, _edges, _edges_g, n_segments):
#     edges_in_frame = _edges[_edges_g[_frame]]
#     best_matches = _get_best_edge_match(edges_in_frame)
        
#     segments_linked = []
#     matched_points = set(best_matches.keys())
#     points_in_frame = _points[[x for c in _points_g[_frame] for x in c]]
#     points_in_frame_g = {x['point_id']:x for x in points_in_frame}
    
#     for seg_id in range(n_segments):
#         points_in_segment = _points[_points_g[_frame][seg_id]]
        
#         prev_inds = {x[-1]['point_id']:x for x in segments_linked}
#         matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
#         matched_indices_cur = [best_matches[x] for x in matched_indices_prev]
        
#         new_points = set(points_in_segment['point_id']) - set(matched_indices_cur)
        
        
#         for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
#             prev_inds[k1].append(points_in_frame_g[k2])
#         segments_linked += [[points_in_frame_g[x]] for x in new_points]
    
#     return segments_linked
