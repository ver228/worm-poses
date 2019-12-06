#!/bin/bash

#$ -P rittscher.prjc -q short.qc -pe shmem 2

source /etc/profile.d/modules.sh
module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

source activate pytorch-1.0
python $HOME/GitLab/worm-poses/collect/resize_and_testtrain.py