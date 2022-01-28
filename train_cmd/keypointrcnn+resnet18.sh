#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


source activate pytorch-1.0

DATADIR='/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/worm-poses/rois4training_filtered'
SAVEDIR=$HOME/worm_models

python -m worm_poses.train \
--n_epochs 1000 \
--data_type 'v2+boxes' \
--model_name 'keypointrcnn+resnet18' \
--loss_type 'maxlikelihood' \
--batch_size 28 \
--roi_size 256 \
--num_workers 4 \
--lr 1e-4 \
--save_frequency 200
