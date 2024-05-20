# compute pre-processings
python main.py --cfg $1 wandb.use False optim.max_epoch 1 &
wait

for SEED in {0..3}; do
    CUDA_VISIBLE_DEVICES=$SEED python main.py --cfg $1 device cuda:0 seed $SEED &
done
wait