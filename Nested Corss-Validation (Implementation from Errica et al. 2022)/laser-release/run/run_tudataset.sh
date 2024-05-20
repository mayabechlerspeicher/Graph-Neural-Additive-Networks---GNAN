for dataset in MUTAG ENZYMES PROTEINS COLLAB REDDIT-BINARY IMDB-BINARY; do
    CUDA_VISIBLE_DEVICES=$2 python main.py --cfg $1 device cuda:0 dataset.name $dataset
done