MODEL_PATH=
SRC_FILE=/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/movies/mating/CB369_PS3398_Set2_Pos5_Ch2_180608_144551.hdf5
SAVEDIR=/Users/avelino/Library/CloudStorage/OneDrive-ImperialCollegeLondon/OXFORD/onedrive_nexus/worms/movies/mating/test

python -m worm_poses.eval \
--model_path $MODEL_PATH \
--src_file $SRC_FILE \
--reader_type 'tierpsy' \
--save_dir $SAVEDIR \
--batch_size 2 \
--images_queue_size 2