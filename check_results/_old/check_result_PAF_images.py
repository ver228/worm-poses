#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import tqdm
import torch
import matplotlib.pylab as plt
import numpy as np
import cv2


from worm_poses.encoders import maps2skels

from check_result_PAF_rois import load_model, get_device



if __name__ == '__main__':
    cuda_id = 0
    
    bn = 'allPAF_PAF+CPM_mse_20190620_003146_adam_lr0.0001_wd0.0_batch48'
    
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/v1' / bn / 'checkpoint.pth.tar'
     
   
    device = get_device(cuda_id)
    model, preeval_func = load_model(model_path)
    model = model.to(device)
    #%%
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC010_v1_images/')
    
    for ifname, fname in enumerate(tqdm.tqdm(list(root_dir.glob('*w2*.tif')))):
        
        img = cv2.imread(str(fname), -1)
        if ifname > 5:
            break
#        plt.figure()
#        plt.imshow(img)
           
        img = (img/4095).astype(np.float32)
        #bot, top = img.min(), img.max()
        #img = (img.astype(np.float32)-bot)/(top-bot)
        #img = cv2.resize(img, dsize=(0,0), fx=1.25, fy=1.25)
        
        X = img[None, None]
        X = torch.tensor(X)
        
        #X = X*255 #i trained the first model using the 0,255 instead of 0, 1.
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            cpm_maps_r, paf_maps_r = outs[-1]
            cpm_maps_r = preeval_func(cpm_maps_r)
            
        
        cpm_map = cpm_maps_r[0].detach().cpu().numpy()
        paf_maps = paf_maps_r[0].detach().cpu().numpy()
        
        skels_pred =  maps2skels(cpm_map, paf_maps, _is_debug = False)
       
        
        mid = 24
        
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        
        
        axs[0].imshow(cpm_map.max(axis=0))
        axs[1].imshow(img, cmap='gray')
        for ss in skels_pred:
            plt.plot(ss[:, 0], ss[:, 1], '.-')
            plt.plot(ss[mid, 0], ss[mid, 1], 'o')
        plt.suptitle(fname.name)     
        
        if ifname > 2:
            break
        #%%
        