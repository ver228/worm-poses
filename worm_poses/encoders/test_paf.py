from encoder import get_skeletons_maps, get_part_affinity_maps
from decoder_paf import maps2skels
from misc import remove_duplicated_annotations, get_best_match

import numpy as np

if __name__ == '__main__':
    from pathlib import Path
    import gzip
    import pickle
    import tqdm
    import matplotlib.pylab as plt
    
    def _paf2draw(paf):
        _paf = paf/2 + 0.5
        _paf = np.concatenate((_paf, _paf[0][None]))
        _paf = np.rollaxis(_paf, 0, 3)
        #assert np.all((_paf >= 0) & (_paf <= 1.))
        return _paf
    
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/worm-poses/rois4training/'
    fname = Path(root_dir) / 'manual_validation.p.zip'
#    root_dir = Path.home() / 'workspace/WormData/worm-poses/rois4training/20190606_174815/'
#    fname = Path(root_dir) / 'manual_train.p.zip'
            
    PAF_seg_dist = 3
    fold_skeleton = True
    _is_debug = False
    
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
            
    #worms2check = data
    worms2check = [x for x in data if x[-1].shape[0]>1]
    
    
    inds2check = range(len(worms2check))
    #inds2check = [7, 18, 39, 43, 69, 83]
    #inds2check = [37]#[110]#[50]#[10]#[50]#[114]#[10]#[17]#[24] #5
    #inds2check = [1301]
    #inds2check = [31]
    
    bad_indeces = []
    for w_ind in tqdm.tqdm(inds2check):
        roi_mask, roi_full, widths, skels = worms2check[w_ind]
        
        roi = roi_mask
        
        n_segments = skels.shape[1]
        midbody_ind = n_segments//2
        
        skel_maps = get_skeletons_maps(skels, widths, roi.shape, fold_skeleton = fold_skeleton)
        affinity_maps = get_part_affinity_maps(skels, widths, roi.shape, PAF_seg_dist, fold_skeleton = fold_skeleton)
        
        #%%
        pred_skels =  maps2skels(skel_maps, affinity_maps, PAF_seg_dist = PAF_seg_dist, _is_debug = _is_debug)
        if len(pred_skels) == len(skels):
            selected_true, closest_dist = get_best_match(pred_skels, skels)
            
            if len(set(selected_true)) == len(selected_true) and all([x < 1. for x in closest_dist]):
                continue
            
            if _is_debug:
                break
        
        #%%
        
        bad_indeces.append(w_ind)
        fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
        
        axs[0][0].imshow(roi_mask, cmap = 'gray')
        for skel in skels:
            axs[0][0].plot(skel[..., 0], skel[..., 1], '.-')
        
        axs[0][1].imshow(roi_mask, cmap = 'gray')
        for skel in pred_skels:
            axs[0][1].plot(skel[..., 0], skel[..., 1], '.-')
            
            
        axs[0][2].imshow(skel_maps[-1], cmap = 'gray')
        
        axs[1][0].imshow(skel_maps[0], cmap = 'gray')
        axs[1][1].imshow(skel_maps[5], cmap = 'gray')
        
        axs[1][2].imshow(_paf2draw(affinity_maps[-1]))
        
        plt.suptitle(w_ind)
        #%%
        
        
        break
    
    