echo ""
echo "****** Hidden dimensions set to 64 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 104 --agg_oper cat
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 64 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 101 --agg_oper cat
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 64 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 116 --agg_oper cat
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 64 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 9 --agg_oper cat
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper cat
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 24 --agg_oper cat
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper cat
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 64 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 110 --agg_oper cat
python -u node_class_sub_feature.py --data film --layer 3 --hidden 64 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 92 --agg_oper cat

echo ""
echo "****** Hidden dimensions set to 128 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 74 --agg_oper cat
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 66 --agg_oper cat
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 33 --agg_oper cat
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 99 --agg_oper cat
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 128 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper cat
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 61 --agg_oper cat
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 128 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper cat
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 82 --agg_oper cat
python -u node_class_sub_feature.py --data film --layer 3 --hidden 128 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 96 --agg_oper cat

echo ""
echo "****** Hidden dimensions set to 256 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 102 --agg_oper cat
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 56 --agg_oper cat
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 75 --agg_oper cat
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 99 --agg_oper cat
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 22 --agg_oper cat
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 256 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper cat
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 61 --agg_oper cat
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 83 --agg_oper cat
python -u node_class_sub_feature.py --data film --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 93 --agg_oper cat

echo ""
echo "****** Hidden dimensions set to 512 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 105 --agg_oper cat
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 93 --agg_oper cat
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 35 --agg_oper cat
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 83 --agg_oper cat
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 512 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 58 --agg_oper cat
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 91 --agg_oper cat
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 100 --agg_oper cat
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 83 --agg_oper cat
python -u node_class_sub_feature.py --data film --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 74 --agg_oper cat
