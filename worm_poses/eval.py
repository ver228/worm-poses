from pathlib import Path
from argparse import ArgumentParser
from .utils import get_device
from .inference import load_model, process_file, read_images_loopbio, read_images_from_tierpsy, unlinked2linked_skeletons


READER_FUNCS = {
    'tierpsy' : read_images_from_tierpsy,
    'loopbio' : read_images_loopbio
}

def process_worm_file(opt):
    device = get_device(opt.cuda_id)
    model = load_model(opt.model_path, device)
    reader_func = READER_FUNCS[opt.reader_type]
    process_args = dict(
        model=model, 
        device=device, 
        batch_size=opt.batch_size, 
        images_queue_size=opt.images_queue_size, 
        reader_func=reader_func
    )
    link_args = dict(
        n_segments = opt.n_segments, 
        fps = opt.fps,
        smooth_window_s = opt.smooth_window_s,
        max_gap_btw_traj_s = opt.max_gap_btw_traj_s,
        target_n_segments = opt.target_n_segments
        max_frac_dist_btw_half = opt.max_frac_dist_btw_half,
        max_frac_dist_btw_seg = opt.max_frac_dist_btw_seg,
        max_frac_dist_btw_skels = opt.max_frac_dist_btw_skels
    )
    
    src_file = Path(opt.src_file)
    save_name = Path(opt.save_dir) / (src_file.stem + '_unlinked-skels.hdf5') 
    process_file(src_file, save_name, **process_args)
    unlinked2linked_skeletons(save_name)

parser = ArgumentParser()
#ARGUMENTS USED FOR THE LANDMARK PREDICTION
parser.add_argument('--model_path', type=str, help='Path to a pretrained model.')
parser.add_argument('--src_file', type=str, help='Path to the video file to be processed.')
parser.add_argument('--reader_type', type=str, choices=['tierpsy', 'loopbio'], default='tierpsy', help='Type of file to be process')
parser.add_argument('--save_dir', type=str, help='Path where the results are going to be saved.')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU number to be used')
parser.add_argument('--batch_size', type=int, default=2, help ='Number of images used per training step')
parser.add_argument('--images_queue_size', type=int, default=2, help ='Number of images used per training step')

#ARGMEUNTS USED ON THE LINKAGE OF TRAJECTORIES
parser.add_argument('--network_n_segments', type=int, default=8, help ='Number of segments output by the neural network prediction.')
parser.add_argument('--final_n_segments', type=int, default= 49, help ='Number of segments produced after linking and smoothing the landmarks.')
parser.add_argument('--fps', type=float, default=25, help ='frames per seconds expected in the video')
parser.add_argument('--smooth_window_s', type=float, default= 0.25, help ='Temporal window used to smooth the skeletons.')
parser.add_argument('--max_gap_btw_traj_s', type=float, default= 0.5, help ='Max temporal gap between frames for a frame to be considered linked')
parser.add_argument('--max_frac_dist_btw_half', type=float, default= 0.5, help = 'Max distance between halves to be considered as the same worm.')
parser.add_argument('--max_frac_dist_btw_seg', type=float, default= 0.125, help = 'Max distance between segments to be considered in the same skeleton half.')
parser.add_arguments('--max_frac_dist_btw_skels', type=float, default= 0.25, help = 'Max distance between skeletons in consecutive half to be considered in the same worm trajectory.')

if __name__ == '__main__':
    opt = parser.parse_args()
    process_worm_file(opt)