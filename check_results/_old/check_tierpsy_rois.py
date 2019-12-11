#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:54:56 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.encoders import maps2skels, extrack_keypoints, join_skeletons_simple


import tables
import pandas as pd
import torch
import math
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
#%%
from check_result_PAF_rois import load_model, get_device

if __name__ == '__main__':
    
    
    #bn = 'allPAF_PAF+CPM_mse_20190618_181029_adam_lr0.0001_wd0.0_batch48'
    bn = 'allPAF_PAF+CPM_maxlikelihood_20190618_183953_adam_lr0.0001_wd0.0_batch48'
    
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/v1' / bn / 'checkpoint.pth.tar'
    
    cuda_id = 1
    device = get_device(cuda_id)
    model, preeval_func = load_model(model_path)
    model = model.to(device)
    #%%
    file2save = Path.home() / 'pose_predictions.pdf'
    
    files2check = [
            ('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch2_01072017_093003.hdf5',
             [(13839, 1), (6236, 1), (20986, 1)]
                    ),
            ('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/ED3017_Ch1_01072017_102004.hdf5',
             [( 8570, 1)]
                    ),
            ('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/mutliworm_example/BRC20067_worms10_food1-10_Set2_Pos5_Ch2_02062017_121709.hdf5',
             [(831, 23),  (14750, 537), (13611, 515), (12443, 444)]
             ),
            ('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5',
             [(1814, 334), (75512, 27014), (75966, 27345), (75966, 27488)]
             )
            
            ]
    
    #fid_pdf = PdfPages(file2save)
    
    for mask_file, rows2check in files2check:
    
        mask_file = Path(mask_file)
        bn = mask_file.stem
        feats_file = mask_file.parent / (bn + '_featuresN.hdf5')
        if not feats_file.exists():
            feats_file = mask_file.parent / 'Results' / (bn + '_featuresN.hdf5')
            assert feats_file.exists()
        
        with pd.HDFStore(feats_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        
    
        
        for frame_number, roi_id in rows2check:
        
            good = trajectories_data['frame_number'] == frame_number
            good &= trajectories_data['worm_index_joined'] == roi_id
            row2read = trajectories_data[good].iloc[0]
            
            with tables.File(mask_file, 'r') as fid:
                img = fid.get_node('/mask')[frame_number]
            
            roi_size = int(row2read['roi_size'])
            ll = int(math.ceil(roi_size/2))
            xl = int(max(row2read['coord_x'] - ll, 0))
            yl = int(max(row2read['coord_y'] - ll, 0))
            roi = img[yl:yl + roi_size, xl:xl + roi_size]
            
            X = torch.from_numpy(roi[None, None]).float()
            X /= 255
            
            with torch.no_grad():
                X = X.to(device)
                outs = model(X)
                cpm_maps_r, paf_maps_r = outs[-1]
                cpm_maps_r = preeval_func(cpm_maps_r)
            
           
            cpm_map = cpm_maps_r[0].detach().cpu().numpy()
            paf_maps = paf_maps_r[0].detach().cpu().numpy()
            
            keypoints = extrack_keypoints(cpm_map, threshold_relative = 0.25, threshold_abs = .25)
            skels_pred = join_skeletons_simple(keypoints, max_edge_dist = 10)
            
            #skels_pred =  maps2skels(cpm_map, paf_maps, _is_debug = False)
            #%%
            fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10, 5))
            axs[0].imshow(roi, cmap='gray', vmin=0, vmax=255)
            axs[0].set_title('Original')
            
            axs[1].imshow(cpm_map.max(axis=0))
            axs[1].set_title('Max Projection Believe Maps')
            
            axs[2].imshow(roi, cmap='gray', vmin=0, vmax=255)
            for ss in skels_pred:
                axs[2].plot(ss[:, 0], ss[:, 1], '-')
                axs[2].plot(ss[25, 0], ss[25, 1], 'o')
            axs[2].set_title('Predictions')
            for ax in axs:
                ax.axis('off')
        
            #fid_pdf.attach_note(f'{mask_file.stem} Frame:{frame_number}') 
            #fid_pdf.savefig()
    #fid_pdf.close()
            