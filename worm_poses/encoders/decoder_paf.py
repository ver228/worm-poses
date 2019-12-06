import numpy as np
import math
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment
import pandas as pd

#%% DECODER
def _line_integral(v1, v2, paf_map, max_edge_dist):
    #%%
    _shift = v2 - v1
    _mag = np.linalg.norm(_shift)
    
    overlap_score = 0.5
    val = 0
    
    if _mag <= max_edge_dist and _mag > 0:
        
        u = _shift / _mag
        
        if u[0] != 0 and u[1] != 0:
            ci = np.arange(0, _shift[0], u[0])
            cj = np.arange(0, _shift[1], u[1])
            
            if len(ci) > len(cj):
                ci = ci[:-1]
            
            if len(cj) > len(ci):
                cj = cj[:-1]
        
        elif u[0] == 0:
            cj = np.arange(0, _shift[1], u[1])
            ci = np.zeros_like(cj)
        else:
            ci = np.arange(0, _shift[0], u[0])
            cj = np.zeros_like(ci)
        
        if len(ci) != len(cj):
            import pdb
            pdb.set_trace()
        
        ci = ci.round().astype(np.int) + v1[0]
        cj = cj.round().astype(np.int) + v1[1]
        
        overlap_score = 0
        if paf_map.shape[0] > 2:
            overlaps = paf_map[2, ci, cj]
            ind = np.argmax(np.abs(overlaps))
            overlap_score = overlaps[ind]
            #shift to a scale from 0 to 1, 0 a crossing, one parallel and 0.5 none
            overlap_score = overlap_score/2 + 0.5
        
        
        L = paf_map[:2, ci, cj]
        u_inv = u[::-1]
        val = np.mean(np.dot(u_inv[None], L))
        
    return val, _mag, overlap_score
#%%
def _get_edges(points_coords, paf_maps, paf_map_dist = 5, max_edge_dist = 50, _is_debug = False):
    
    
    edges_vals = []
    for t1, paf_map in enumerate(paf_maps):
        
        t2 = t1 + paf_map_dist
        points1 = points_coords[t1]
        points2 = points_coords[t2]
        
        if _is_debug:
            import matplotlib.pylab as plt
            plt.figure()
            plt.imshow(paf_map[0])
            plt.plot(points1[..., 1], points1[..., 0], 'vr')
            plt.plot(points2[..., 1], points2[..., 0], 'or')
        
        if len(points1) == 0 or len(points2) == 0:
            continue
        
        M = np.zeros((len(points1), len(points2)), np.float32)
        D = np.full(M.shape, max_edge_dist)
        
        for i1, v1 in enumerate(points1):
            for i2, v2 in enumerate(points2):
                val, segment_magnitude, _ = _line_integral(v1, v2, paf_map, max_edge_dist)
                
                M[i1, i2] = max(0, val)
                D[i1, i2] = segment_magnitude
                
        #convert maximization to a minimization problem
        cost_matrix = (M.max() - M) + (D / (D.min(axis=0) + 1.))
        
        if cost_matrix.shape[0] < cost_matrix.shape[1]:
            cost_matrix = cost_matrix.T
        
        #I am adding a second column that cost the double just to allow for cases 
        #where it might be convenient to add to the same edge
        cost_matrix = np.concatenate((cost_matrix, cost_matrix*2), axis=1)
        
        try:
            col_ind, row_ind = linear_sum_assignment(cost_matrix)
        except:
            import pdb
            pdb.set_trace()
            
        if M.shape[0] < M.shape[1]:
            row_ind, col_ind = col_ind, row_ind
        
        row_ind = row_ind % M.shape[1]
        col_ind = col_ind % M.shape[0]
        
        for ind1, ind2 in zip(col_ind, row_ind):
            if D[ind1, ind2] > max_edge_dist:
                continue
            
            row = (t1, ind1, t2, ind2, M[ind1, ind2], D[ind1, ind2])
            edges_vals.append(row)
            
            if _is_debug:
                y, x = zip(points1[ind1], points2[ind2])
                
                plt.plot(x,y, 'r')
            
        
    edges_vals = pd.DataFrame(edges_vals, columns = ['map1', 'ind1', 'map2', 'ind2', 'val', 'size'])  
    
    return edges_vals
 

def _build_skels_halfs(edges_vals_df, points_coords, paf_map_dist, max_edge_dist, min_num_segments = 20, _is_debug = False):
    
    #I making the strong assumtion that most of the maps will have identified a correct number of edges
    expected_n_halfs = math.floor(edges_vals_df['map1'].value_counts().median())
    
    
    edges_g = edges_vals_df.groupby('map1')
    all_edges = []
    for map1, dat in  edges_g:
        edges = []
        for irow, row in dat.iterrows():
            p1 = points_coords[int(row[0])][int(row[1])]
            p2 = points_coords[int(row[2])][int(row[3])]
            
            
            edges.append(np.array((p1, p2)))
        all_edges.append(edges)
    
    
    def _get_cost(seeds, edges, prev_directions, seed_ind = -1):
        #seed_ind -1 means to use the last index in the skeleton
        cost_next = []
        for s, prev_disp in zip(seeds, prev_directions):
            c = []
            for e in edges:
                s_cur = s[seed_ind]
                #distance between edges
                
                
                new_disp = e - s_cur
                mags = np.linalg.norm(new_disp, axis=1)
                
                avg_mag = mags.mean()
                
                new_disp = (new_disp/mags[:, None]).mean(axis=0)
                
                
                if prev_disp is not None and not np.isnan(new_disp).any():
                    #I am adding a sort of inertial penalty. Calculate the direction where the previous segments where added
                    #a segment will have a lower penalty if it is being added in the same direction i.e. if the segments where being
                    #added upwards we want to penalize a segment that moves the curve downwards
                    direction = np.dot(new_disp, prev_disp)
                    score_d = (1 - direction)*5
                    
                else:
                    score_d = 5
                    
                #print(f'L {avg_mag:.2f}, {score_d:.2f}')
                val = avg_mag + score_d
                
                if val != val:
                    import pdb
                    pdb.set_trace()
                
                c.append(val)
            cost_next.append(c)
        cost_next = np.array(cost_next)
        
        
        cost_same = np.full((len(seeds), len(edges)), max_edge_dist, np.float32)
        
        cost_matrix = np.concatenate((cost_next, cost_same), axis=1)
        
        return cost_matrix
    
    
    for ini_ind, edges in  enumerate(all_edges):
        if len(edges) == expected_n_halfs:
            seeds = [[x] for x in edges]
            break
    else:
        return []
    
    prev_directions = [None for _ in seeds]
    for edges in all_edges[ini_ind+1:]:
        if _is_debug:
            import matplotlib.pylab as plt
            plt.figure()
            plt.imshow(roi, cmap = 'gray')
            for ii, ss in enumerate(seeds):
                ss = np.concatenate(ss)
                plt.plot(ss[:, 1], ss[:, 0], '.-')
                
                plt.text(ss[0, 1], ss[0, 0], str(ii), color = 'k')
                
            for e in edges:
                plt.plot(e[:, 1], e[:, 0], '.-c')
                
        
        
        cost_matrix = _get_cost(seeds, edges, prev_directions)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if c < len(edges) and cost_matrix[r, c] < max_edge_dist:
                seeds[r].append(edges[c])
                #if r+1 > len(prev_directions):
                #    prev_directions[r] = None
        
        curr_directions = []
        for s, acc in zip(seeds, prev_directions):
            if len(s) >= 2:
                prev_disp = (s[-1] + 1e-3) - s[-2] 
                mags = np.linalg.norm(prev_disp, axis=1)[:, None]
                prev_disp = prev_disp/mags
                prev_disp = np.mean(prev_disp, axis=0)
                
                if acc is not None:
                    prev_disp = 0.5*acc +  prev_disp*0.5 #add inertial
                curr_directions.append(prev_disp)
            else:
                curr_directions.append(None)
        prev_directions = curr_directions
        assert len(prev_directions) == len(seeds)  
        
       
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
        for ii, ss in enumerate(seeds):
            ss = np.concatenate(ss)
            plt.plot(ss[:, 1], ss[:, 0], '.-')
            
            plt.text(ss[0, 1], ss[0, 0], str(ii), color = 'k')
        
        
    skel_halfs = []
    for ss in seeds:
        half_ = [x[0] for x in ss] + [x[1] for x in ss[-paf_map_dist:]]
        skel_halfs.append(half_)
    skel_halfs = [x for x in skel_halfs if len(x) > min_num_segments]
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(roi, cmap = 'gray')
        for ii, ss in enumerate(skel_halfs):
            ss = np.array(ss)
            plt.plot(ss[:, 1], ss[:, 0])
            plt.text(ss[0, 1], ss[0, 0], str(ii), color = 'k')
        
    return skel_halfs
#%%
def _link_skeletons(points_coords, paf_maps, paf_map_dist = 3, max_edge_dist = 50, _is_debug = False):
    
    max_edge_size = paf_map_dist*max_edge_dist
    edges_vals_df = _get_edges(points_coords, 
                               paf_maps[:-1], 
                               paf_map_dist = paf_map_dist,
                               max_edge_dist = max_edge_dist,
                               _is_debug = _is_debug)
    edges_vals_df = edges_vals_df[edges_vals_df['size']<=max_edge_size]
    
    if len(edges_vals_df) == 0:
        return []
    
    skel_halfs = _build_skels_halfs(edges_vals_df, points_coords, paf_map_dist, max_edge_dist,  _is_debug = _is_debug)
    
    if not len(skel_halfs):
        return []
    
    #join halfs using the affinity field info...
    mid_ind = paf_map_dist//2 + 1
    mid_coords = [x[-mid_ind] for x in skel_halfs]
    paf_map = paf_maps[-1]
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(paf_map[1])
        for ss, (x,y) in enumerate(mid_coords):
            plt.plot(y,x,  '.r')
            plt.text(y,x, str(ss), color = 'r')
    
    n_candidates = len(mid_coords)
    M = np.zeros((n_candidates, n_candidates), np.float32)
    D = np.full(M.shape, max_edge_dist)
    
    for i1 in range(n_candidates):
        for i2 in range(i1 + 1, n_candidates):
            v1 = mid_coords[i1]
            v2 = mid_coords[i2]
            val, segment_magnitude, _ = _line_integral(v1, v2, paf_map, max_edge_dist)

            
            val = val if val > 0 else -val
            
            M[i1, i2] = val
            D[i1, i2] = segment_magnitude
            M[i2, i1] = val
            D[i2, i1] = segment_magnitude
            
    #convert maximization to a minimization problem. I am adding an extra term to penalize distance
    cost_matrix = (M.max() - M)  + D/(D.min() + 1) 
    
    col_ind, row_ind = linear_sum_assignment(cost_matrix)
    matches = [(r,c) if c > r else (c,r) for r,c in zip(row_ind, col_ind)]
    matches = set(matches)
    
    #if the matches are not symetrical i will use the distance to eliminate matches that do not make sense.
    #at most i want to form half of the skeletons i obtained in skel_halfs
    max2keep = math.ceil(len(skel_halfs)/2)
    if len(matches) > max2keep:
        dists = []
        for i1, i2 in matches:
            p1, p2 = map(np.array, (skel_halfs[i1], skel_halfs[i2]))
            dists.append(np.linalg.norm((mid_coords[i1] - mid_coords[i2])))
        matches = [x for _,x in sorted(zip(dists,matches))] #sort using the distances
        matches = matches[:max2keep]
    
    skels = []
    for i1, i2 in matches:
        skel = np.concatenate((skel_halfs[i1][:-1], skel_halfs[i2][::-1]))
        
        
        skels.append(skel)
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(roi, cmap='gray')
        for ss in skels:
            plt.plot(ss[:, 1], ss[:, 0])
    
    skels = [x[:, ::-1] for x in skels] 
    

    return skels


def _get_peaks(cpm_maps, threshold_relative, threshold_abs):
    
    all_coords = []
    for mm in cpm_maps:
        th = max(mm.max()*threshold_relative, threshold_abs)
        coords = peak_local_max(mm, threshold_abs = th)
        
        
        all_coords.append(coords)
        
    return all_coords  

def maps2skels(cpm_maps, paf_maps, PAF_seg_dist=5, _is_debug = False):
    
    points_coords = _get_peaks(cpm_maps, threshold_relative = 0.1, threshold_abs = 0.05)
    
    roi_skels = _link_skeletons(points_coords, paf_maps, paf_map_dist = PAF_seg_dist, _is_debug = _is_debug)
    
    return roi_skels