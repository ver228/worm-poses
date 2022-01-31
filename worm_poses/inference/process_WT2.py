import numpy as np
import tables
import torch
import pandas as pd
from collections import defaultdict
from .process_base import init_reader
from .readers import init_reader, read_images_from_tierpsy

def _get_roi_limits(skel_preds, roi_size, img_lims, corner = None):
    cm = [np.median(p[:, -2:], axis=0).round().astype(np.int) for p in skel_preds]
    cm = np.mean(cm, axis=0)  
    
    xl, yl = cm - roi_size//2
    
    if corner is not None:
        xl += corner[0]
        yl += corner[1]
    
    
    xl = int(min(max(0, xl), img_lims[1]-roi_size))
    yl = int(min(max(0, yl), img_lims[0]-roi_size))
    xr = xl + roi_size
    yr = yl + roi_size
    
    
    return (yl, yr), (xl, xr)

#%%
def predictions2skeletons(preds, 
                          n_segments = 8,
                          min_PAF = 0.25,
                          is_skel_half = True,
                          _frame2test = -1
                          ):
    
    edges_cost = preds['edges_costs']
    edges_indeces = preds['edges_indices']
    points = preds['skeletons']
    
    
    #get the best matches per point
    PAF_cost = edges_cost[0]
    valid = PAF_cost >= min_PAF
    PAF_cost = PAF_cost[valid]
    edges_indeces = edges_indeces[:, valid]
    
    inds = np.argsort(PAF_cost)[::-1]
    edges_indeces = edges_indeces[:, inds]
    _, valid_index =  np.unique(edges_indeces[0], return_index = True )
    best_matches = {x[0]:x[1] for x in edges_indeces[:, valid_index].T}
    matched_points = set(best_matches.keys())
    
    segments_linked = []
    
    #add the point_id column 
    assert (edges_indeces < len(points)).all()
    points =  np.concatenate((np.arange(len(points))[:, None], points), axis=1)     
    for ipoint in range(n_segments):
        points_in_segment = points[points[:, 1] == ipoint]
        
        prev_inds = {x[-1][0]:x for x in segments_linked}
        
        
        
        matched_indices_prev = list(set(prev_inds.keys()) & matched_points)
        matched_indices_cur = [best_matches[x] for x in matched_indices_prev]
    
        new_points = set(points_in_segment[:, 0]) - set(matched_indices_cur)
    
        try:
            for k1, k2 in zip(matched_indices_prev, matched_indices_cur):
                prev_inds[k1].append(points[k2])
        except:
            import pdb
            pdb.set_trace()
        segments_linked += [[points[x]] for x in new_points]
    
    if is_skel_half:
        skeletons = []
        matched_halves = defaultdict(list)
        for _half in segments_linked:
            if len(_half) == n_segments:
                midbody_ind = _half[-1][0]
                matched_halves[midbody_ind].append(_half)
        
        for k, _halves in matched_halves.items():
            if len(_halves) == 2:
                skel = np.array(_halves[0] + _halves[1][-2::-1])
                skel = skel[:, -2:]
                
                skeletons.append(skel)
    else:
        skeletons = [np.array(x)[:, -2:] for x in segments_linked]   
    
        
    return skeletons
#%%
    
class predictionsAccumulator():
    def __init__(self):
        self.edges = []
        self.points = []
        self.current_poind_id = 0
        
    def add(self, frames, predictions, corner = None):
        for frame, preds in zip(frames, predictions):
            #predictions = {k : v.cpu().numpy() for k,v in predictions.items()}
            
            seg_id, skel_x, skel_y = preds['skeletons'].T
            nms_score = preds['scores_abs'].T
            edge_p1, edge_p2 = preds['edges_indices']
            edge_cost_PAF, edge_cost_R = preds['edges_costs']
            
            if corner is not None:
                skel_x = skel_x + corner[0]
                skel_y = skel_y + corner[1]
                
            N = len(seg_id)
            t_p = np.full(N, frame)
            p_ids = range(self.current_poind_id, self.current_poind_id + N)
            self.points += list(zip(p_ids, t_p, seg_id, skel_x, skel_y, nms_score))
            
            t_e = np.full(len(edge_p1), frame)
            edge_p1 = edge_p1 + self.current_poind_id
            edge_p2 = edge_p2 + self.current_poind_id
            self.edges += list(zip(t_e, edge_p1, edge_p2, edge_cost_PAF, edge_cost_R))
            
            self.current_poind_id += N
    

    def save(self, save_name):
        points_df = pd.DataFrame(self.points, columns = ['point_id', 'frame_number', 'segment_id', 'x', 'y', 'score'])
        edges_df = pd.DataFrame(self.edges, columns = ['frame_number', 'point1', 'point2', 'cost_PAF', 'cost_R'])
        
        points_rec = points_df.to_records(index = False)
        edges_rec = edges_df.to_records(index = False)
        
        
        TABLE_FILTERS = tables.Filters(
                            complevel=5,
                            complib='zlib',
                            shuffle=True,
                            fletcher32=True)
        
        with tables.File(save_name, 'w') as fid:
            fid.create_table('/', 'points', obj = points_rec, filters = TABLE_FILTERS)
            fid.create_table('/', 'edges', obj = edges_rec, filters = TABLE_FILTERS)

#%%
def process_from_reader_single(queue_images, save_name, model, device, roi_size=256, full_batch_size = 2 ):
    
    pred_acc = predictionsAccumulator()
    
    def skelfuncs(x, _frame2test):
        return predictions2skeletons(x, 
                                    _frame2test = _frame2test,
                                     n_segments = model.n_segments,
                                     is_skel_half = True
                                     )
    
    roi_limits = None
    with torch.no_grad():
        while True:
            try:
                dat = queue_images.get(timeout = 600)
            except (ConnectionResetError, FileNotFoundError):
                break
            
            if dat is None:
                 break
             
            frames, batch = dat
            
            img_lims = batch.shape[-2:]
            while (roi_limits is None) and (len(batch) > 0):
            
                X = batch[:full_batch_size]
                frames_in_batch = frames[:full_batch_size]
                
                batch = batch[full_batch_size:]
                frames = frames[full_batch_size:]
                
                X = X.to(device)
                full_predictions = model(X)
                full_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in full_predictions]
                
                pred_acc.add(frames_in_batch, full_predictions)
                
                
                
                skels = [skelfuncs(p, _frame2test = frame) for p, frame in zip(full_predictions, frames_in_batch)]
                skels = [s[0] for s in skels if len(s) == 1]
                
                
                if skels:
                    roi_limits = _get_roi_limits(skels, roi_size, img_lims)
                    break
            
            if len(batch) == 0:
                continue
            
            (yl, yr), (xl, xr) = roi_limits
            X_roi = batch[..., yl:yr, xl:xr]
            X_roi = X_roi.to(device)
            roi_predictions = model(X_roi)
            roi_predictions = [{k : v.detach().cpu().numpy() for k,v in p.items()} for p in roi_predictions]
            
            _corner = (xl, yl)
            pred_acc.add(frames, roi_predictions, corner = _corner)
            preds = [x['skeletons'] for x in roi_predictions[-10:]]
            roi_limits = _get_roi_limits(preds, roi_size, img_lims, corner = _corner)
    
           
    pred_acc.save(save_name)

def process_WT2_file(mask_file, save_name, model, device, batch_size, roi_size=256, images_queue_size = 4):
    reader_p, queue_images = init_reader(read_images_from_tierpsy, mask_file, batch_size, device, images_queue_size)
    try:
        process_from_reader_single(queue_images, save_name, model, device, roi_size=roi_size)
    except Exception as e:
        reader_p.terminate()
        raise e