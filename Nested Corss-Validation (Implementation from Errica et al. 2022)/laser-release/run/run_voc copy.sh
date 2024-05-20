# for file in configs/Vocsuperpixels/*; do

#     # compute pre-processings
#     python main.py --cfg $file wandb.use False optim.max_epoch 1 &
#     wait

#     for SEED in {0..3}; do
#         CUDA_VISIBLE_DEVICES=$SEED python main.py --cfg $file device cuda:0 seed $SEED &
#     done
#     wait

# done

CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/Vocsuperpixels/laser.yaml dynamic.additions_factor 0.1 dynamic.num_snapshots 2 &
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/Vocsuperpixels/laser.yaml dynamic.additions_factor 1 dynamic.num_snapshots 2 &
CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/Vocsuperpixels/laser.yaml dynamic.additions_factor 0.5 dynamic_num_snapshots 5 gnn.dim_inner 70&