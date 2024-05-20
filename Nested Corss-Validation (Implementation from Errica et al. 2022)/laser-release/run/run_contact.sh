# compute pre-processings
for config in configs/Contact/*; do
    for SEED in {0..0}; do
        python main.py --cfg $config device cuda:$SEED seed $SEED wandb.use False optim.max_epoch 1
    done
done

for config in configs/Contact/*; do
    for SEED in {0..3}; do
        CUDA_VISIBLE_DEVICES=$SEED python main.py --cfg $config device cuda:0 seed $SEED &
    done
    wait
done