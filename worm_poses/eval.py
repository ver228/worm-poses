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
    
    src_file = Path(opt.src_file)
    save_name = Path(opt.save_dir) / (src_file.stem + '_unlinked-skels.hdf5') 
    process_file(src_file, save_name, **process_args)
    unlinked2linked_skeletons(save_name)

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to a pretrained model.')
parser.add_argument('--src_file', type=str, help='Path to the video file to be processed.')
parser.add_argument('--reader_type', type=str, choices=['tierpsy', 'loopbio'], default='tierpsy', help='Type of file to be process')
parser.add_argument('--save_dir', type=str, help='Path where the results are going to be saved.')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU number to be used')
parser.add_argument('--batch_size', type=int, default=2, help ='Number of images used per training step')
parser.add_argument('--images_queue_size', type=int, default=2, help ='Number of images used per training step')

if __name__ == '__main__':
    opt = parser.parse_args()
    process_worm_file(opt)