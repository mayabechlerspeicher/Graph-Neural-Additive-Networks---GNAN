echo ""
echo "****** No ReLU, Hops:3, hidden = 256 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 103 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 256 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 66 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 256 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 35 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 256 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 9 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 61 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 256 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 84 --agg_oper sum --is_relu 0
python -u node_class_sub_feature.py --data film --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 5 --agg_oper sum --is_relu 0
