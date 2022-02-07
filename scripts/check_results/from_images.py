#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

from pathlib import Path 
import cv2
import torch
import tqdm
import matplotlib.pylab as plt
import numpy as np

from worm_poses.utils import get_device
from worm_poses.inference import load_model, link_segments_single_frame

def process_single_image(model, fname, scale_factor=1, n_segments=8):
    img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/img.max()
    X = torch.from_numpy(img).float()[None, None]
    
    if scale_factor != 1.:
        #rescale the image if it is too different from the expected size from the training set
        X = torch.nn.functional.interpolate(X, scale_factor = scale_factor)
    predictions, belive_maps = model(X)
    
    predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
    
    segments = link_segments_single_frame(predictions[0], n_segments=n_segments)
    segments = [np.stack(x) for x in segments]
    
    skels = predictions[0]['skeletons'].astype(np.float32)
    skels[:, -2:] /= scale_factor 
    
    cpm, paf = belive_maps
    cpm = cpm[0].detach().numpy()
    cpm_max = cpm.max(axis = 0)
    
    paf = paf[0].detach().numpy()
    paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
    
    fig, axs = plt.subplots(1,3, figsize = (30, 10), sharex = True, sharey = True)
    for ax in axs:
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
    for ii in range(model.n_segments):
        valid = skels[..., 0] == ii
        axs[1].plot(skels[valid, -2], skels[valid, -1], '.')
    
    for seg in segments:
        seg = seg[:, -2:]/ scale_factor
        axs[2].plot(seg[:, -2], seg[:, -1], '-')
        
def process_from_images(model_path, fnames, cuda_id=0, scale_factor = 1., n_segments = 8):
    """[summary]

    Args:
        model_path ([type]): [Path to the model weights]. 
        fnames ([type]): [list of paths to be processed.]
        cuda_id (int, optional): [gpu id to be used]. Defaults to 0.
        scale_factor ([type], optional): [description]. Defaults to 1..
        n_segments (int, optional): [description]. Defaults to 8.
    """    

    device = get_device(cuda_id)
    extra_args = {'return_belive_maps' : True}
    model = load_model(model_path, device, extra_args=extra_args)
    model.eval()
    
    
    for ifname, fname in enumerate(tqdm.tqdm(fnames)):
        process_single_image(model, fname, scale_factor=scale_factor, n_segments=n_segments)

if __name__ == '__main__':
    model_path = Path('/Users/avelino/Downloads/Save_model/v5/v5_openpose_maxlikelihood_20220131_160257/model_best.pth.tar')
    root_dir = Path('/Users/avelino/Downloads/Images')
    fnames = list(root_dir.glob('*png'))[:1]
    
    process_from_images(model_path, fnames)
