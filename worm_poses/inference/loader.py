from typing import Union
from pathlib import Path
import torch
from worm_poses.train import get_model_from_args
from worm_poses.configs import read_model_config

def load_model(model_path : Union[str, Path], device):
    #I am assuming the model type is a prefix on the subfolder containing the model
    #TODO in a future i should read the configuration file directly.
    model_path = Path(model_path)
    basename = model_path.parent.name
    model_name = basename.split('_')[1]
    
    conf_args = read_model_config(model_name)
    model = get_model_from_args(conf_args)

    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)

    return model