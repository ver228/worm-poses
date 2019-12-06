import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from worm_poses.models import CPM_PAF, PretrainedBackBone, get_preeval_func

from worm_poses.encoders import remove_duplicated_annotations, get_best_match, resample_curve
from worm_poses.encoders.decoder_simple import extrack_keypoints, join_skeletons_simple


import torch
import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform



def load_model(model_path):
    
    n_segments = 25
    n_affinity_maps = 21
    bn = model_path.parent.name
    
    preeval_func = get_preeval_func(bn)
    
    if 'vgg19' in bn:
        backbone = PretrainedBackBone('vgg19')
    elif 'resnet50' in bn:
        backbone = PretrainedBackBone('resnet50')
    else:
        backbone = None
    
    is_PAF = 'PAF' in bn
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True,
                             backbone = backbone,
                             is_PAF = is_PAF
                             )
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, preeval_func

def get_device(cuda_id):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    return device

def maps2skeletons_simple_debug(belive_maps, max_edge_dist = 10):
    
    #key_points are going to be sorted as [Nx3] where the last dimension is the score...
    keypoints = []
    
    for belive_map in belive_maps:
        #belive_map = cv2.blur(belive_map, (3,3))
        map_coords = cv2_peak_local_max(belive_map, threshold_relative = 0.25, threshold_abs = .25)
        #if map_coords.size == 0:
        #    continue
        
        map_scores = belive_map[map_coords[:, 0], map_coords[:, 1]]
        res = np.concatenate((map_coords, map_scores[:, None]), axis=1)
        keypoints.append(res)
        
        if _is_debug:
            
            plt.figure()
            plt.imshow(belive_map, cmap='gray')
            plt.plot(map_coords[:, 1], map_coords[:, 0], 'r.')
            plt.suptitle(w_ind)
    
    #the edges are head/tail to midbody. I want to invert them...
    keypoints = keypoints[::-1]
    n_expected_halfs = np.median([len(x) for x in keypoints[1:]]) # I am skiping the midbody that should have half of the points
    n_expected_halfs = math.floor(n_expected_halfs)
    n_expected_halfs = n_expected_halfs if n_expected_halfs%2 == 0 else n_expected_halfs + 1
    
    if len(keypoints[0]) == n_expected_halfs//2:
        #if the mid
        mid_bodies = [x for x in keypoints[0]]
        seeds = [[x] for x in mid_bodies + mid_bodies]
        ini_seg = 1
    else:
        for ii in range(1, len(keypoints)//4):
            if len(keypoints[ii]) == n_expected_halfs:
                seeds = [[x] for x in keypoints[ii]]
                break
        else:
            fig, axs = plt.subplots(2,2, sharex= True, sharey= True)
            axs = axs.flatten()
            
            axs[0].imshow(roi, cmap = 'gray')
            for ss in skels:
                axs[0].plot(ss[:, 0], ss[:, 1])
            
            
            axs[1].imshow(belive_maps.max(axis=0), cmap = 'gray')
            bad_indeces.append(w_ind)
            
        
    
    for next_keypoints in keypoints[ini_seg:]:
        
        cost_matrix = []
        for seed in seeds:
            ss = seed[-1]
            
            seed_coords = ss[:2]
            
            delta = seed_coords[None] - next_keypoints[:, :2]
            dist = np.sqrt((delta**2).sum(axis=1))
            cost_matrix.append(dist)
        cost_matrix = np.array(cost_matrix)
        
        
        #I am adding a second column that cost the double just to allow for cases 
        #where it might be convenient to repeat the lower dimenssion
        cost_matrix = np.concatenate((cost_matrix, cost_matrix*10), axis=0)
        
        
        
        cost_matrix[cost_matrix>max_edge_dist] = max_edge_dist
        
        
        
        col_ind, row_ind = linear_sum_assignment(cost_matrix)
        
        good = cost_matrix[col_ind, row_ind] < max_edge_dist
        col_ind, row_ind = col_ind[good], row_ind[good]
        
        for c, r in zip(col_ind, row_ind):
            if c < len(seeds) and r < len(next_keypoints):
                seeds[c].append(next_keypoints[r])
       
         
        if _is_debug:
            plt.figure()
            plt.imshow(roi, cmap = 'gray')
            #plt.imshow(max_believe, cmap = 'gray')
            for ii, ss in enumerate(seeds):
                ss = np.array(ss)
                plt.plot(ss[:, 1], ss[:, 0])
                plt.text(ss[-1, 1], ss[-1, 0], str(ii), color = 'k')
            plt.plot(next_keypoints[:, 1], next_keypoints[:, 0], '.r')
        
        
    skel_halfs = [np.array(x) for x in seeds]
    if _is_debug:
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
        for ii, ss in enumerate(skel_halfs):
            ss = np.array(ss)
            plt.plot(ss[:, 1], ss[:, 0])
            plt.text(ss[0, 1], ss[0, 0], str(ii), color = 'k')
            
    
    #%%
    # join halves
    mid_coords = [x[0] for x in skel_halfs]
    
    n_candidates = len(mid_coords)
    
    if n_candidates == 2:
        matches = [(0, 1)] #there is only one pair, nothing to do here
    else:
        dist = pdist(mid_coords)
        dist = squareform(dist)
        np.fill_diagonal(dist, np.max(dist))
        col_ind, row_ind = linear_sum_assignment(dist)
        matches = set([(c,r) if c > r else (r, c) for c,r in zip(col_ind, row_ind)])
    
    skels_pred = []
    for i1, i2 in matches:
        h1, h2 = skel_halfs[i1][::-1, :2], skel_halfs[i2][:, :2]
        
        skel = np.concatenate((h1[:, ::-1], h2[:, ::-1]))
        skels_pred.append(skel)
    
    return skels_pred, keypoints
    
if __name__ == '__main__':
    from pathlib import Path
    import gzip
    import pickle
    import tqdm
    import matplotlib.pylab as plt
    
    cuda_id = 0
    _is_debug = False
    
    bn = 'allPAF_PAF+CPM_maxlikelihood_20190620_003150_adam_lr0.0001_wd0.0_batch48'
    model_path = Path.home() / 'OneDrive - Nexus365/worms/worm-poses/models' / bn / 'checkpoint-499.pth.tar'#'checkpoint.pth.tar'

    #bn = 'v1_CPM_maxlikelihood_20190622_223955_adam_lr0.0001_wd0.0_batch96'
    #bn = 'allPAF_PAF+CPM_mse_20190620_003146_adam_lr0.0001_wd0.0_batch48'
    model_path = Path.home() / 'workspace/WormData/worm-poses/results/v1' / bn / 'checkpoint.pth.tar'#'checkpoint-499.pth.tar'

    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    fname = Path(root_dir) / 'manual_validation.p.zip'
    #fname = Path(root_dir) / 'from_tierpsy_validation.p.zip'
    
    
    
    #root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190606_174815/'
    #fname = Path(root_dir) / 'manual_train.p.zip'
    #fname = Path(root_dir) / 'from_tierpsy_train.p.zip'
    
    with gzip.GzipFile(fname, 'rb') as fid:
        data_raw = pickle.load(fid)
        
    data = []
    for roi_mask, roi_full, widths, skels in data_raw:
        if np.isnan(skels).any():
            continue
        if roi_mask.sum() == 0:
            continue
        skels, widths = remove_duplicated_annotations(skels, widths)
        
        data.append((roi_mask, roi_full, widths, skels))
            
    worms2check = data
    #worms2check = [x for x in data if x[-1].shape[0]>1]
    worms2check = [x for x in data if x[-1].shape[0]==1]
    #%%
    
    device = get_device(cuda_id)
    model, preeval_func = load_model(model_path)
    model = model.to(device)
    
    
    inds2check = range(len(worms2check))
    #inds2check = [7, 18, 39, 43, 69, 83]
    #inds2check = [37]#[110]#[50]#[10]#[50]#[114]#[10]#[17]#[24] #5
    #inds2check = [1301]
    #inds2check = [31]
    #inds2check = [3, 5]
    
    #inds2check = [427]#[479]#[360]#
    #inds2check = [317]#[498]
    roi_size = 96
    n_segments = 49
    #%%
    
    bad_indeces = []
    for w_ind in tqdm.tqdm(inds2check):
        roi_mask, roi_full, widths, skels = [x.copy() if x is not None else None for x in worms2check[w_ind]]
        roi = roi_mask
        
        l_max = max(roi.shape)
        
        scale_f = roi_size/l_max
        roi = cv2.resize(roi, dsize = (0,0), fx = scale_f, fy = scale_f)
        assert max(roi.shape) <= roi_size
        skels *= scale_f
        
        n_segments = skels.shape[1]
        midbody_ind = n_segments//2
        
        X = torch.from_numpy(roi).float()
        X /= 255
        X = X[None, None]
        with torch.no_grad():
            X = X.to(device)
            outs = model(X)
            
            if len(outs[-1]) == 2:
                belive_maps_r, paf_maps_r = outs[-1]
            else:
                belive_maps_r = outs[-1]
            belive_maps_r = preeval_func(belive_maps_r)
            belive_maps_r = belive_maps_r.detach().cpu().numpy()
            
        belive_maps = belive_maps_r[0]
        
        
        keypoints = extrack_keypoints(belive_maps, threshold_relative = 0.25, threshold_abs = .25)
        skels_pred = join_skeletons_simple(keypoints, max_edge_dist = 10)
        
        skels_pred = [resample_curve(skel, n_segments) for skel in skels_pred]
        selected_true, closest_dist = get_best_match(skels_pred, skels)
    
        #skels_pred = [x[:, :2] for x in skel_halfs]
        #%%
        if len(set(selected_true)) != len(skels) or max(closest_dist) > 3.:
            bad_indeces.append(w_ind)
            
            fig, axs = plt.subplots(2,2, sharex= True, sharey= True)
            axs = axs.flatten()
            
            axs[0].imshow(roi, cmap = 'gray')
            for ss in skels:
                axs[0].plot(ss[:, 0], ss[:, 1])
            
            
            axs[1].imshow(belive_maps.max(axis=0), cmap = 'gray')
            
            axs[2].imshow(roi, cmap = 'gray')
            for cc in keypoints:
                axs[2].plot(cc[:, 1], cc[:, 0], '.r')
            
            axs[2].imshow(roi, cmap = 'gray')
            for cc in keypoints:
                axs[2].plot(cc[:, 1], cc[:, 0], '.r')
            
            axs[3].imshow(roi, cmap = 'gray')
            for ss in skels_pred:
                axs[3].plot(ss[:, 0], ss[:, 1])
            plt.suptitle(w_ind)
        
            
    #%%
    
    print(bn, 1 - len(bad_indeces)/len(inds2check))
        
              
