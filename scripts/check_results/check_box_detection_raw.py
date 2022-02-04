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
import matplotlib.pylab as plt
import numpy as np
from matplotlib import patches

from worm_poses.utils import get_device
from worm_poses.inference import load_model

def process_single_img(model, img):
    img = img.astype(np.float32)/255.
        
    #X = torch.from_numpy(np.repeat(img[None], 3, axis=0)).float()[None]
    X = torch.from_numpy(img[None]).float()[None]
    
    predictions = model(X)
    
    fig_loc, ax = plt.subplots(1,1)
    ax.imshow(img, cmap='gray')
    
    preds = predictions[0]
    for lab, bbox in zip(preds['labels'], preds['boxes']):
        bbox = bbox.detach().numpy()
        
        xmin, ymin, xmax, ymax = bbox
        ww = xmax - xmin + 1
        hh = ymax - ymin + 1
        rect = patches.Rectangle((xmin, ymin),ww,hh,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        s = str(lab.item())
        ax.text(x, y, s, color='cyan', fontsize=12)
        
    fig_keypoints = None
    if 'keypoints' in preds:
        maps = preds['keypoints']
        bboxes = preds['boxes']
        for mm, bbox in zip(maps, bboxes):
            N, W, H = mm.shape
            
            mm_s = torch.nn.functional.softmax(mm.view(N, -1), dim = 1)
            mm_s = mm_s.view(N, W, H)
            
            mm_s = mm_s.cpu().detach().numpy()
            
            mm_max = mm_s.max(0)
            xmin, ymin, xmax, ymax = bbox.round().int().numpy()
            roi = img[ ymin:ymax, xmin:xmax]
            
            
            fig_keypoints, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(roi)
            axs[1].imshow(mm_max)
    
    return fig_loc, fig_keypoints

def process_w_box_detection(model_path, mask_file, cuda_id = 0, frame2read = 0, field2read = '/full_data'):
    """[summary]
    Load an image from Tierpsy mask file and show the results of pose detection model using mask rcnn as backbone.
    Args:
        model_path ([type]): [Path to the model weights]. 
        mask_file ([type]): [Path to tierpsy mask file.]
        cuda_id (int, optional): [gpu id to be used]. Defaults to 0.
        frame2read (int, optional): [frame number in the mask file to be read]. Defaults to 0.
        field2read (str, optional): [field in the mask file to be read]. Defaults to '/full_data'.
    """    
    device = get_device(cuda_id)
    model = load_model(model_path, device)
    model.eval()
    
    mask_file = Path(mask_file)
    with tables.File(mask_file, 'r') as fid:
        img = fid.get_node(field2read)[frame2read]
    process_single_img(model, img)

if __name__ == '__main__':
    model_path = ''
    mask_file = ''
    process_w_box_detection(model_path, mask_file)
    
    
    