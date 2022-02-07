#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:05:51 2019

@author: avelinojaver
"""

from pathlib import Path 
import torch
import numpy as np
import random
import tqdm
import gzip
import pickle
import matplotlib.pylab as plt
import numpy as np

from worm_poses.utils import get_device
from worm_poses.inference import load_model, link_segments_single_frame
    
def process_single_img(model, roi, skels_true, scale_factor=1., n_segments=8):
    X = roi.astype(np.float32)/255
    X = torch.from_numpy(X).float()[None, None]
    
    
    if scale_factor != 1.:
        #rescale the image if it is too different from the expected size from the training set
        X = torch.nn.functional.interpolate(X, scale_factor = scale_factor)
    
    predictions, belive_maps = model(X)
    predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
    
    segments = link_segments_single_frame(predictions[0], n_segments=n_segments)
    segments = [np.stack(x) for x in segments]
    
    p = predictions[0]['skeletons']
    p = p[p[:, 0] < n_segments]
    
    scores_abs = predictions[0]['scores_abs']
    
    fig, axs = plt.subplots(2, 4, figsize = (20, 20), sharex = True, sharey = True)
    for ax in axs.flatten():
        ax.imshow(roi, cmap = 'gray')
        ax.axis('off')
        
    for skel in skels_true:
        axs[0][1].plot(skel[:, 0], skel[:, 1], '.')
    
    
    axs[0][2].scatter(p[:, -2], p[:, -1],  c = p[:, 0], cmap = 'jet')
    for seg in segments:
        delx = (random.random()-0.5)*1e-0
        dely = (random.random()-0.5)*1e-0
        axs[0][3].plot(seg[:, -2] + delx, seg[:, -1] + dely, '.-')
    
    cpm, paf = belive_maps
    cpm = cpm[0].detach().numpy()
    cpm_max = cpm[:n_segments].max(axis = 0)
    
    paf = paf[0].detach().numpy()
    paf_max = np.linalg.norm(paf, axis = 1).max(axis=0)
    
    axs[1][0].imshow(cpm_max)
    axs[1][1].imshow(paf_max)
    axs[1][2].imshow(np.linalg.norm(paf[0], axis = 0))
    axs[1][3].imshow(np.linalg.norm(paf[-1], axis = 0))
    return fig


def process_from_validation_data(model_path, validation_set_path, inds2process, cuda_id=0, scale_factor = 1., n_segments = 8):
    """[summary]

    Args:
        model_path ([type]): [Path to the model weights]. 
        validation_set_path ([type]): [Path to validation/test file in the same format as the training data.]
        inds2process ([type]): [indeces to be processed. the training file is a list of images, this variable select what images are going to be processed.]
        cuda_id (int, optional): [gpu id to be used]. Defaults to 0.
        scale_factor ([type], optional): [description]. Defaults to 1..
        n_segments (int, optional): [description]. Defaults to 8.
    """    
    device = get_device(cuda_id)
    extra_args = {'return_belive_maps' : True}
    model = load_model(model_path, device, extra_args=extra_args)
    model.eval()
    

    with gzip.GzipFile(validation_set_path, 'rb') as fid:
        data_raw = pickle.load(fid)
    
    model.nms_min_distance = 1
    
    dat = [data_raw[ii] for ii in inds2process]
    for ii, out in enumerate(tqdm.tqdm(dat)):
        roi = out[1] if out[1] is not None else out[0]
        skels_true = out[3]
        if skels_true.shape[0] < 1:
            continue
        process_single_img(model, 
                            roi, 
                            skels_true, 
                            scale_factor=scale_factor, 
                            n_segments=n_segments)
        

if __name__ == '__main__':
    model_path = Path.home() / 'workspace/WormData/worm-poses/results' / set_type  / bn / 'model_best.pth.tar'
    validation_set_path = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training_filtered/manual_test.p.zip'
    inds2process = [722, 721, 719, 717, 49, 48, 44, 41, 21, 17, 10]
    process_from_validation_data(model_path, validation_set_path, inds2process)