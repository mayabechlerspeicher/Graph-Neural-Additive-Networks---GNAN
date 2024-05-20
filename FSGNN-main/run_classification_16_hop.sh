python -u node_class.py --data cora --layer 16 --w_att 0.1 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.02 --layer_norm 1 --dev 0
python -u node_class.py --data citeseer --layer 16 --w_att 0.001 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0
python -u node_class.py --data pubmed --layer 16 --w_att 0.0 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0
python -u node_class.py --data chameleon --layer 16 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --lr_att 0.04 --layer_norm 1 --dev 0
python -u node_class.py --data wisconsin --layer 16 --w_att 0.0 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --lr_att 0.01 --layer_norm 1 --dev 0
python -u node_class.py --data texas --layer 16 --w_att 0.01 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.005 --layer_norm 1 --dev 0
python -u node_class.py --data cornell --layer 16 --w_att 0.01 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --lr_att 0.04 --layer_norm 1 --dev 0
python -u node_class.py --data squirrel --layer 16 --w_att 0.1 --w_fc2 0.0 --w_fc1 0.0 --dropout 0.6 --lr_fc 0.005 --lr_att 0.02 --layer_norm 1 --dev 0
python -u node_class.py --data film --layer 16 --w_att 0.0001 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --lr_att 0.04 --layer_norm 1 --dev 0
echo "=== Actor dataset with no hop normalization ==="
python -u node_class.py --data film --layer 16 --w_att 0.01 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.005 --lr_att 0.005 --layer_norm 0 --dev 0

