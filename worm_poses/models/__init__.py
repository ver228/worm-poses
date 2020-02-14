import os
from pathlib import Path
pretrained_path = Path.home() / 'workspace/pytorch/pretrained_models/'

try:
    if pretrained_path.exists():
        os.environ['TORCH_HOME'] = str(pretrained_path)
except OSError:
    pass

from .pose_detector import PoseDetector
from .fasterrcnn import get_keypointrcnn