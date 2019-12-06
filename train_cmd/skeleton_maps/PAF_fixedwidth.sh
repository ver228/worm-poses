#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q@compG004 -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-0.4.1

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

SCRIPTPATH="$HOME/GitLab/worm-poses/scripts/train_PAF.py"
#SCRIPTPATH="`pwd`/GitLab/worm-poses/scripts/train_PAF.py"

python -W ignore $SCRIPTPATH --n_epochs 1000 --data_type 'all-fixed-width' \
--n_segments 25 --n_affinity_maps 20 --batch_size 16 --num_workers 16 \
--lr 0.0001 --save_frequency 25

echo "Finished at :"`date`
exit 0