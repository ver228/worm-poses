DATADIR='/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/worm-poses/rois4training_filtered'
SAVEDIR=$HOME/worm_models

python -m worm_poses.train \
--n_epochs 3000 \
--data_type 'v5' \
--data_dir $DATADIR \
--save_dir $SAVEDIR \
--model_name 'openpose+light+head' \
--loss_type 'maxlikelihood' \
--batch_size 24 \
--num_workers 8 \
--lr 1e-4 \
--save_frequency 600
