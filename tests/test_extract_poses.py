#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:11:35 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_poses.models import CPM_PAF

from scipy.interpolate import interp1d

import torch
import tables
import cv2
import math
import numpy as np
from pathlib import Path

from test_simple_extract_poses import _get_peaks, _link_skeletons



def _get_device(cuda_id = 0):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device

#%%
from threading import Thread
from queue import Queue

class ThreadedGenerator(object):
    """
    https://gist.github.com/everilae/9697228
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator,
                 sentinel = object(),
                 queue_maxsize = 1,
                 daemon = False,
                 Thread = Thread,
                 Queue = Queue):
        
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                self._queue.put(value)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()



#%%
TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='zlib',
    shuffle=True,
    fletcher32=True)
#%%
def image_reader(mask_file, frames2check = None):
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        
        if frames2check is None:
            frames2check = range(masks.shape[0])
        
        for frame in tqdm.tqdm(frames2check, desc = 'Frames readed'):
            img = masks[frame]
            yield frame, img

def img2rois(gen, batch_size, roi_size_base):    
    roi_batches = {}
    for frame_number, img in gen:
        img[:16, :480] = 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4)
        
        for ss in stats[1:]:
            left, top, width, height, area = ss
            roi = img[top:top + height, left:left + width]
            
            roi_size = math.ceil(max(x/roi_size_base for x in roi.shape))*roi_size_base
            
            #TODO... what happends near the image edges?
            _pad = [roi_size - x for x in roi.shape]
            _pad = [(math.ceil(x/2), math.floor(x/2)) for x in _pad]
            
            roi = np.pad(roi, _pad, 'constant')
            
            
            row = roi, (frame_number, ss, _pad)
            
            if roi_size not in roi_batches:
                roi_batches[roi_size] = []
            
            roi_batches[roi_size].append(row)
            
        
        _roi_batches = {}
        for r, v in roi_batches.items():
            _batch_size = math.ceil(batch_size*(roi_size_base/r)**2)
            _batch_size = int(max(1, _batch_size))
            if len(v) >= _batch_size:
                yield v[:_batch_size]
                _roi_batches[r] = v[_batch_size:]
            else:
                _roi_batches[r] = v
        roi_batches = _roi_batches
    
    for r, v in roi_batches.items():
        if len(v) > 0:
            yield v
            
def _rois2maps(X, model, device):
    with torch.no_grad():
        X = torch.tensor([x[None] for x in X]).float()
        X = X.to(device)
        X /= 255.
        outs = model(X)
    cpm_maps, paf_maps = outs[-1]
    
    cpm_maps = cpm_maps.detach().cpu().numpy()
    paf_maps = paf_maps.detach().cpu().numpy()
    return cpm_maps, paf_maps

def rois2maps(gen, model, device):
    for batch in gen:
        rois, rois_data = zip(*batch)
        b_cpm_maps, b_paf_maps = _rois2maps(rois, model, device)
        yield rois_data, b_cpm_maps, b_paf_maps

def maps2skels(gen, min_skel_size = 40):
    for rois_data, b_cpm_maps, b_paf_maps in gen:
        for roi_data, cpm_maps, paf_maps in zip(rois_data, b_cpm_maps, b_paf_maps):
            frame_number, stat, pad_size = roi_data
            left, top, width, height, area = stat
            
            (xl, xr), (yl, yr) = pad_size
            
            xr = cpm_maps.shape[1]-xr
            yr = cpm_maps.shape[2]-yr
            
            cpm_maps = cpm_maps[:, xl:xr, yl:yr]
            paf_maps = paf_maps[:, :, xl:xr, yl:yr]
            
            if not (cpm_maps.shape[1] == height and cpm_maps.shape[2] == width):
                print(cpm_maps.shape, paf_maps.shape)
                raise ValueError
            
            
            all_coords = _get_peaks(cpm_maps, threshold_relative = 0.5, threshold_abs = 0.05)
            paf_maps_inv = paf_maps[::-1]
            roi_skels = _link_skeletons(all_coords, paf_maps_inv, paf_map_dist = 5)
        
            
            corner = np.array((left, top))[None]
            skels = [ss[:, ::-1] + corner for ss in roi_skels]
            
            for ss in skels:
                if ss.shape[0] > min_skel_size: 
                    yield frame_number, ss

def resample_curve(curve, resampling_N=49):
    '''Resample curve to have resampling_N equidistant segments'''

    # calculate the cumulative length for each segment in the curve
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)

    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]

    # Verify array lengths
    if len(lengths) < 2 or len(curve) < 2:
        return None, None, None

    fx = interp1d(lengths, curve[:, 0])
    fy = interp1d(lengths, curve[:, 1])

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    # I add the epsilon because otherwise the interpolation will produce nan
    # for zero
    try:
        resampled_curve = np.zeros((resampling_N, 2))
        resampled_curve[:, 0] = fx(subLengths)
        resampled_curve[:, 1] = fy(subLengths)
    except ValueError:
        resampled_curve = np.full((resampling_N, 2), np.nan)

    return resampled_curve
#%%
if __name__ == '__main__': 
    import tqdm
    _ext = 'skelNN.hdf5'
    
#    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
#    mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5'
#    mask_file = Path(mask_file)
#    skel_file = mask_file.parent / (mask_file.stem + _ext)
#    _model_path = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/trained_models/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'
#
#    
    
    #mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
    #mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase3/MaskedVideos/wildMating1.1_CB4856_self_CB4856_self_PC3_Ch1_15082018_114818.hdf5'
    #mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase1/MaskedVideos/N2_worms10_CSCD068947_10_Set2_Pos5_Ch1_08082017_212337.hdf5'
    mask_file = Path.home() / 'workspace/WormData/results/worm-poses/raw/Phase2/MaskedVideos/JU2587_worms10_food1-10_Set1_Pos4_Ch1_20102017_125044.hdf5'
    
    skel_file = Path.home() / 'workspace/WormData/results/worm-poses/test_out/Phase3/' / (mask_file.stem + _ext)
    _model_path = Path.home() / 'workspace/WormData/results/worm-poses/logs/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'

    
    n_segments = 25
    n_affinity_maps = 20
    
    batch_size_base = 128
    roi_size_base = 80
    
    cuda_id = 3
    #%%
    is_short = False
    
    if is_short:
        frames2check = list(range(100))#list(range(16100, 16150))
        
    else:
        frames2check = None
        
    device = _get_device(cuda_id)
    
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True)
    
    state = torch.load(_model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    model = model.to(device)
    model.eval()
    #%%
    gen = image_reader(mask_file, frames2check = frames2check)
    gen = img2rois(gen, batch_size_base, roi_size_base)
    gen = ThreadedGenerator(gen)
    
    gen = rois2maps(gen, model, device)
    gen = ThreadedGenerator(gen)
    
    gen = maps2skels(gen)
    
    
    skel_file.parent.mkdir(exist_ok = True, parents = True)
    
    skeleton_id = 0    
    with tables.File(str(skel_file), 'w') as fid:
        
        tab_dtypes = np.dtype([('frame_number', np.int), ('skeleton_id' , np.int)])
        inds_tab = fid.create_table('/',
                    "skel_info",
                    tab_dtypes,
                    filters = TABLE_FILTERS)
        
        skeletons = fid.create_earray('/', 
                        'skeletons',
                        atom = tables.Float32Atom(),
                        shape = (0, 49, 2),
                        chunkshape = (1, 49, 2),
                        filters = TABLE_FILTERS
                        )
        
        for (frame, skel) in tqdm.tqdm(gen, 'Skeletons processed'):
            inds_tab.append([(frame, skeleton_id)])
            
            skel = resample_curve(skel, resampling_N=49)
            
            skeletons.append(skel[None])
            skeleton_id += 1

