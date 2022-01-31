DATADIR='/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/worm-poses/rois4training_filtered'
SAVEDIR=$HOME/worm_models

python -m worm_poses.train \
--data_dir $DATADIR \
--save_dir $SAVEDIR \
--n_epochs 1000 \
--flow_config 'v2' \
--model_config 'keypointrcnn+resnet18' \
--loss_type 'maxlikelihood' \
--batch_size 28 \
--num_workers 4 \
--lr 1e-4 \
--save_frequency 200
