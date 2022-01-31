import tables
import numpy as np
import torch
import tqdm
import time
from pathlib import Path
import multiprocessing as mp
import imgstore


def read_images_from_tierpsy(mask_file, batch_size, queue):
    bn = Path(mask_file).stem
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        tot = masks.shape[0]
        
        for frame_number in tqdm.trange(0, tot, batch_size, desc = bn):
            X = masks[frame_number:frame_number + batch_size]
            X = X.astype(np.float32)/255.
            X = torch.from_numpy(X).unsqueeze(1)
            frames = list(range(frame_number, frame_number + X.shape[0]))
            queue.put((frames, X))
            
    queue.put(None)
    
    while not queue.empty():
        #wait until the consumer empty the queue before destroying the process.
        time.sleep(1)

def _prepare_batch(batch):
    frames, X = map(np.array, zip(*batch))
    X = X.astype(np.float32)/255.
    X = torch.from_numpy(X).unsqueeze(1)
    return frames, X

def read_images_loopbio(file_name, batch_size, queue):
    store = imgstore.new_for_filename(str(file_name))
    bn = Path(file_name).parent.name
    
    
    batch = []
    for frame_number in tqdm.trange(store.frame_count, desc = bn):
        img = store.get_next_image()[0]
        batch.append((frame_number, img))
        if len(batch) >= batch_size:
            frames, X = _prepare_batch(batch)
            batch = []
            queue.put((frames, X))
    
    if len(batch) > 0:
        frames, X = _prepare_batch(batch)
        queue.put((frames, X))
    
            
    queue.put(None)
    
    while not queue.empty():
        #wait until the consumer empty the queue before destroying the process.
        time.sleep(1)

def init_reader(target_func, mask_file, batch_size, images_queue_size):
    queue_images = mp.Queue(images_queue_size)
    
    reader_p = mp.Process(target = target_func, 
                          args= (mask_file, batch_size, queue_images)
                          )
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process
    return reader_p, queue_images
    