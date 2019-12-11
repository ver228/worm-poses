import os
from pathlib import Path
pretrained_path = Path.home() / 'workspace/pytorch/pretrained_models/'
if pretrained_path.exists():
    os.environ['TORCH_HOME'] = str(pretrained_path)

from .pose_detector import PoseDetector
from .fasterrcnn import get_keypointrcnn