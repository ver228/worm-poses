#!/bin/bash

#$ -P rittscher.prjb -q short.qb
#$ -t 1-12196

export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

FILESSOURCE=$HOME/workspace/files2link.txt
echo "Username: " `whoami`
FSOURCE=$(awk "NR==$SGE_TASK_ID" $FILESSOURCE)
echo $FSOURCE

source activate pytorch-1.0
cd $HOME/GitLab/worm-poses/process/link_skeletons
python --version
python link_file.py "$HOME/$FSOURCE"

