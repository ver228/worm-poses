# worm-poses

## Installation

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
