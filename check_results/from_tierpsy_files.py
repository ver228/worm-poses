#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
import tables
import torch
from worm_poses.models import PoseDetector
import tqdm

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import numpy as np
    
    #bn = 'v2_openpose+light_maxlikelihood_20191210_172128_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v2_openpose+light_maxlikelihood_20191211_150642_adam_lr0.0001_wd0.0_batch32'
    
    #bn = 'v2_openpose+head_maxlikelihood_20191219_150412_adam_lr0.0001_wd0.0_batch20'
    #bn = 'v2_openpose+light+full_maxlikelihood_20191226_114337_adam_lr0.0001_wd0.0_batch32'
    
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191228_165156_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_094906_adam_lr0.0001_wd0.0_batch22'
    #bn = 'v2_openpose+light+fullsym_maxlikelihood_20191229_223132_adam_lr0.0001_wd0.0_batch22'
    
    #bn = 'v2_openpose+light+head_maxlikelihood_20191219_144819_adam_lr0.0001_wd0.0_batch32'
    #bn = 'v3_R+openpose+light+head_maxlikelihood_20200117_183236_adam_lr0.0001_wd0.0_batch20'
    
    #bn = 'v3_openpose+light+head_maxlikelihood_20200118_100732_adam_lr0.0001_wd0.0_batch24'
    
    #bn = 'v4_openpose+light+head_maxlikelihood_20200129_104454_adam_lr0.0001_wd0.0_batch24'
    #bn = 'v3_openpose+light+head_maxlikelihood_20200129_080810_adam_lr0.0001_wd0.0_batch24'
    
    bn = 'v4PAFflat_openpose+light+head_maxlikelihood_20200206_105708_adam_lr0.0001_wd0.0_batch24'
    
    set_type = bn.partition('_')[0]
    
    #model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    model_path = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/models/' / bn / 'model_best.pth.tar'
    #'checkpoint.pth.tar'
    
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    if 'openpose+light+fullsym' in bn:
        model_args = dict(
            n_segments = 15,
            n_affinity_maps = 14,
            features_type = 'vgg11',
            n_stages = 4,
            )
        
    elif 'openpose+light' in bn:
        model_args = dict(
            n_segments = 8,
            n_affinity_maps = 8,
            features_type = 'vgg11',
            n_stages = 4,
            )
    else:
        model_args = dict(
            n_segments = 25,
            n_affinity_maps = 21,
            features_type = 'vgg19',
            n_stages = 6,
            )
        
    if '+head' in bn:
        model_args['n_segments'] += 1
        
    #%%
    model = PoseDetector(**model_args, return_belive_maps = True)
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    # fnames = ['/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5',
    #           '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    #           ]
    
    # fnames = [Path.home() / 'workspace/WormData/screenings/mating_videos/Mating_Assay/Mating_Assay_030718/MaskedVideos/Set1_CB369_CB1490/Set1_CB369_CB1490_Ch1_03072018_163429.hdf5',
    #          Path.home() / 'workspace/WormData/screenings/mating_videos/wildMating/MaskedVideos/20180819_wildMating/wildMating5.2_MY23_self_CB4856_self_PC2_Ch1_19082018_123257.hdf5',
    #          Path.home() / 'workspace/WormData/screenings/mating_videos/wildMating/MaskedVideos/20180818_wildMating/wildMating4.2_CB4856_self_CB4856_self_PC3_Ch2_18082018_124339.hdf5'
    #          ]
    
    # fnames = ['/Users/avelinojaver/workspace/WormData/screenings/pesticides_adam/Syngenta/MaskedVideos/test_SYN_001_Agar_Screening_310317/N2_worms10_food1-3_Set2_Pos4_Ch5_31032017_220113.hdf5',
    #           '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    #           ]
    
    #fnames = ['/Users/avelinojaver/OneDrive - Nexus365/worms/movies/Serena_WT_Screening/15.1_5_cx11314_da_cb4852_ff_Set0_Pos0_Ch1_03032018_145825.hdf5']
    
    fnames = ['/Users/avelinojaver/OneDrive - Nexus365/worms/movies/hydra_example_for_Avelino/MaskedVideos//syngenta_screen_run1_prestim_20191214_151023.22956809/metadata.hdf5',
              '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/hydra_example_for_Avelino/MaskedVideos//syngenta_screen_run1_poststim_20191214_152134.22956809/metadata.hdf5',
              '/Users/avelinojaver/OneDrive - Nexus365/worms/movies/hydra_example_for_Avelino/MaskedVideos//syngenta_screen_run1_bluelight_20191214_151529.22956809/metadata.hdf5'
              ]
    #%%
    for mask_file in tqdm.tqdm(fnames[:]):
        
        mask_file = Path(mask_file)
        with tables.File(mask_file, 'r') as fid:
            img = fid.get_node('/full_data')[0]
            #img = fid.get_node('/mask')[11107]
        
        #img = img.astype(np.float32)/255.
        img = img.astype(np.float32)/img.max()
        #img = img[512:-512, 512:-512]
        
        with torch.no_grad():
            X = torch.from_numpy(img).float()[None, None]
            predictions, belive_maps = model(X)
            
            #skels = predictions['skeletons'].detach().cpu().numpy()
            skels = predictions[0]['skeletons'].detach().cpu().numpy()
            
            cpm, paf = belive_maps
            cpm = cpm[0].detach().numpy()
            cpm_max = cpm.max(axis = 0)
            
            paf = paf[0].detach().numpy()
            paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(img, cmap='gray')
        for ii in range(model.n_segments):
            valid = skels[..., 0] == ii
            plt.plot(skels[valid, -2], skels[valid, -1], '.')
        
        
        fig, axs = plt.subplots(1,5, figsize = (30, 5), sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(cpm[0])
        axs[2].imshow(np.linalg.norm(paf[0], axis = 0))
        #axs[3].imshow(cpm[-3])
        #axs[4].imshow(np.linalg.norm(paf[-3], axis = 0))
        axs[3].imshow(cpm[-1])
        axs[4].imshow(np.linalg.norm(paf[-1], axis = 0))
        #axs[3].imshow(cpm_max)
        #axs[4].imshow(paf_max)

            
            