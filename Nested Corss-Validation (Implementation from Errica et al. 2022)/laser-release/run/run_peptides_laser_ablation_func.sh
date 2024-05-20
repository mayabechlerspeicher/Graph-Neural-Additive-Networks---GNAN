# compute pre-processings
for i in "2 270" "3 250" "4 235" "5 225"; do
    set -- $i
    # $1 is number snapshots and $2 is hidden dim
    for density in 0.1 0.25 0.5 1.0; do
        for config in configs/Peptides-laserglobal-ablation/peptides-func-GCN-laserglobal.yaml; do
            python main.py --cfg $config wandb.use False optim.max_epoch 1 dynamic.num_snapshots $1 dynamic.additions_factor $density gnn.dim_inner $2 &
        done
    done
    wait
done

for i in "2 270" "3 250" "4 235" "5 225"; do
    set -- $i
    # $1 is number snapshots and $2 is hidden dim
    for density in 0.1 0.25 0.5 1.0; do
        for config in configs/Peptides-laserglobal-ablation/peptides-func-GCN-laserglobal.yaml; do
            for SEED in {0..3}; do
                CUDA_VISIBLE_DEVICES=$SEED python main.py --cfg $config device cuda:0 seed $SEED dynamic.num_snapshots $1 dynamic.additions_factor $density gnn.dim_inner $2 &
            done
            wait
        done
    done
done