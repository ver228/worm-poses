import yaml
from pathlib import Path

_WD = Path(__file__).resolve().parent
def _read_config_file(config_name, config_type):
    fname = _WD / config_type / (config_name + '.yaml')
    if not fname.exists():
        raise ValueError(f'Config[{config_type}] : `{config_name}` does not exists.')

    with open(fname) as fid:
        config = yaml.safe_load(fid)
    return config


def read_flow_config(config_name):
    return _read_config_file(config_name, 'data')


def read_model_config(config_name):
    return _read_config_file(config_name, 'models')
