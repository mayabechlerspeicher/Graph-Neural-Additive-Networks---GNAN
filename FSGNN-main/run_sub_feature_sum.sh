echo ""
echo "****** Hidden dimensions set to 64 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 72 --agg_oper sum 
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 64 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 113 --agg_oper sum 
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 64 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 100 --agg_oper sum 
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 64 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 4 --agg_oper sum 
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 64 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data film --layer 3 --hidden 64 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 3 --agg_oper sum 

echo ""
echo "****** Hidden dimensions set to 128 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 117 --agg_oper sum 
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 113 --agg_oper sum 
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 128 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 64 --agg_oper sum 
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 128 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 128 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data film --layer 3 --hidden 128 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 25 --agg_oper sum

echo ""
echo "****** Hidden dimensions set to 256 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 105 --agg_oper sum 
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 66 --agg_oper sum 
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 5 --agg_oper sum 
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 2 --agg_oper sum 
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 256 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 16 --agg_oper sum 
python -u node_class_sub_feature.py --data film --layer 3 --hidden 256 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 104 --agg_oper sum

echo ""
echo "****** Hidden dimensions set to 512 ******"
python -u node_class_sub_feature.py --data cora --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 105 --agg_oper sum 
python -u node_class_sub_feature.py --data citeseer --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 27 --agg_oper sum 
python -u node_class_sub_feature.py --data pubmed --layer 3 --hidden 512 --w_fc2 0.0001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 25 --agg_oper sum 
python -u node_class_sub_feature.py --data chameleon --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 7 --agg_oper sum 
python -u node_class_sub_feature.py --data wisconsin --layer 3 --hidden 512 --w_fc2 0.0 --w_fc1 0.001 --dropout 0.6 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 24 --agg_oper sum 
python -u node_class_sub_feature.py --data texas --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.005 --layer_norm 1 --dev 0 --hop_idx 0 --agg_oper sum 
python -u node_class_sub_feature.py --data cornell --layer 3 --hidden 512 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 24 --agg_oper sum 
python -u node_class_sub_feature.py --data squirrel --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.0 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 16 --agg_oper sum 
python -u node_class_sub_feature.py --data film --layer 3 --hidden 512 --w_fc2 0.001 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hop_idx 99 --agg_oper sum 
