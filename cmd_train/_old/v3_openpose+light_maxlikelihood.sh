#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


source activate pytorch-1.0

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

SCRIPTPATH="$HOME/GitLab/worm-poses/scripts/train_PAF.py" 
python -W ignore $SCRIPTPATH \
--n_epochs 3000 \
--data_type 'v3' \
--model_name 'openpose+light' \
--loss_type 'maxlikelihood' \
--batch_size 24 \
--num_workers 4 \
--lr 1e-4 \
--save_frequency 500

echo "Finished at :"`date`
exit 0