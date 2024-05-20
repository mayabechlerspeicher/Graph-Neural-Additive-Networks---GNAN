for config in configs/FOSR/*; do
    for SEED in {0..3}; do
        python main.py --cfg $config device cuda:$SEED seed $SEED &
    done
    wait
done