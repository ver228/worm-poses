#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q@compG004 -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-0.4.1

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

SCRIPTPATH="$HOME/GitLab/worm-poses/scripts/train_model.py"
python $SCRIPTPATH --n_epochs 1000 --model_name 'CPM' --batch_size 32 --num_workers 1 --lr 0.0001  --copy_tmp '/tmp/avelino'

echo "Finished at :"`date`
exit 0