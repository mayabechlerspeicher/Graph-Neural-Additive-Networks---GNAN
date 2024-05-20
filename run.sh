#!/bin/bash

data_name=pubmed
seed=0
num_epochs=1000
wd=0
normalize_m=1
dropout=0.6
lr=0.01
readout_n_layers=0
n_layers=2
hidden_channels=64
is_for_plot=0
processed_data_dir=processed_data

python main.py --seed=$seed --wd=$wd --model_name=gnam --data_name=$data_name --run_grid_search=0 --dropout=$dropout --readout_n_layers=$readout_n_layers  --n_layers=$n_layers --hidden_channels=$hidden_channels  --lr=$lr --num_epochs=$num_epochs --early_stop=0 --limited_m=0 --normalize_m=$normalize_m --bias=1 --is_for_plot=$is_for_plot --processed_data_dir=$processed_data_dir
