cuda=0
for dataset in MUTAG ENZYMES PROTEINS REDDIT-BINARY IMDB-BINARY COLLAB; do
    for config in configs/TUDataset-baselines/*; do
        CUDA_VISIBLE_DEVICES=$(($cuda%4)) python main.py --cfg $config device cuda:0 dataset.name $dataset &
        let cuda=$cuda+1
        if [[ $(($cuda % 4)) -eq 0 ]]; then
            wait
        fi
    done
done