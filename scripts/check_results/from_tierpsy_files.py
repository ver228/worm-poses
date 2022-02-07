#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

from pathlib import Path 
import tables
import torch
import matplotlib.pylab as plt
import numpy as np
    
from worm_poses.utils import get_device
from worm_poses.inference import load_model

def process_w_box_detection(model_path, mask_file, cuda_id = 0, frame2read = 0, field2read = '/full_data'):
    """[summary]
    Load an image from Tierpsy mask file and show the results of pose detection based on the openpose architecture.
    Args:
        model_path ([type]): [Path to the model weights]. 
        mask_file ([type]): [Path to tierpsy mask file.]
        cuda_id (int, optional): [gpu id to be used]. Defaults to 0.
        frame2read (int, optional): [frame number in the mask file to be read]. Defaults to 0.
        field2read (str, optional): [field in the mask file to be read]. Defaults to '/full_data'.
    """   
     
    device = get_device(cuda_id)
    extra_args = {'return_belive_maps' : True}
    model = load_model(model_path, device, extra_args=extra_args)
    model.eval()
    
    mask_file = Path(mask_file)
    with tables.File(mask_file, 'r') as fid:
        img = fid.get_node(field2read)[frame2read]
        
    img = img.astype(np.float32)/img.max()
    
    with torch.no_grad():
        X = torch.from_numpy(img).float()[None, None]
        predictions, belive_maps = model(X)
        
        skels = predictions[0]['skeletons'].detach().cpu().numpy()
        
        cpm, paf = belive_maps
        cpm = cpm[0].detach().numpy()
        cpm_max = cpm.max(axis = 0)
        
        paf = paf[0].detach().numpy()
        paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
    
    fig_skels, ax = plt.subplots(1,1)
    ax.imshow(img, cmap='gray')
    for ii in range(model.n_segments):
        valid = skels[..., 0] == ii
        plt.plot(skels[valid, -2], skels[valid, -1], '.')
    
    
    fig_paf, axs = plt.subplots(1,5, figsize = (30, 5), sharex = True, sharey = True)
    axs[0].imshow(img, cmap = 'gray')
    axs[1].imshow(cpm[0])
    axs[2].imshow(np.linalg.norm(paf[0], axis = 0))
    #axs[3].imshow(cpm[-3])
    #axs[4].imshow(np.linalg.norm(paf[-3], axis = 0))
    axs[3].imshow(cpm[-1])
    axs[4].imshow(np.linalg.norm(paf[-1], axis = 0))
    #axs[3].imshow(cpm_max)
    #axs[4].imshow(paf_max)
    return fig_skels, fig_paf
            
if __name__ == '__main__':
    model_path = ''
    mask_file = ''
    process_w_box_detection(model_path, mask_file)