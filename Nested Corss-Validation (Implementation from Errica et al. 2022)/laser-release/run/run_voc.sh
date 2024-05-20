for file in configs/Vocsuperpixels/*; do

    # compute pre-processings
    python main.py --cfg $file wandb.use False optim.max_epoch 1 &
    wait

    for SEED in {0..3}; do
        CUDA_VISIBLE_DEVICES=$SEED python main.py --cfg $file device cuda:0 seed $SEED &
    done
    wait

done