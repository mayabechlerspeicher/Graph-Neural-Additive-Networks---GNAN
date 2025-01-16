#!/bin/bash

data_name=mutagenicity
model_name=gnan
seed=0
num_epochs=1000
wd=0
dropout=0.6
lr=0.01
n_layers=3
hidden_channels=64

python main.py --seed=$seed --wd=$wd --model_name=$model_name --data_name=$data_name --dropout=$dropout  --n_layers=$n_layers --hidden_channels=$hidden_channels  --lr=$lr --num_epochs=$num_epochs
