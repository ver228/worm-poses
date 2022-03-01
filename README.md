# worm-poses

## Installation

Clone this repo and inside it execute.
```
pip install -r requirements.txt
pip install -e .
```

The dataset needed to train the model is here:
https://imperiallondon-my.sharepoint.com/:f:/g/personal/ajaver_ic_ac_uk/EjGxbc9w6yRKl1KFmfgUq8oBexPgxzZRfyCEhs9Yda_L4A?e=O4fnZW

but you need an Imperial College account to access to it.

## Training

Once the model is repository you can execute, where DATADIR is the path to the training data, and SAVEDIR is the path where the model is going to be saved.

```
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
```

There are more examples of training configs in the folder `train_cmd`.



## Prediction Examples

![Example1](https://user-images.githubusercontent.com/8364368/156197436-e231d19a-6387-4dcb-b6bd-f7715bcad870.png)


![Example2](https://user-images.githubusercontent.com/8364368/156197733-6e71c764-f6c3-44ac-8d78-3c4d0ebc3ed3.png)



## Evaluation on worm movies

You can evaluated the models on tierpsy movies (https://github.com/Tierpsy/tierpsy-tracker) as

```
python -m worm_poses.eval \
--model_path $MODEL_PATH \
--src_file $SRC_FILE \
--reader_type 'tierpsy' \
--save_dir $SAVEDIR \
--batch_size 2 \
--images_queue_size 2
```

Where MODEL_PATH is the path for a pretrained model, SRC_FILE is the path to source video file, SAVEDIR is where the results are going to be saved.

