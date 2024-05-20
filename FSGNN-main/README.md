# FSGNN
Implementation of FSGNN. For more details, please refer to [our paper](https://arxiv.org/abs/2105.07634).
This work is further extended into our [second](https://arxiv.org/abs/2111.06748) paper.
Experiments were conducted with following setup:  
Pytorch: 1.6.0  
Python: 3.8.5  
Cuda: 10.2.89
Trained on NVIDIA V100 GPU.

**Summary of results**

| **Dataset** | **3-hop Accuracy(%)** | **8-hop Accuracy(%)** | **16-hop Accuracy(%)** | **32-hop Accuracy(%)** |
| :-------------- | :---------------------: | :---------------------: | :--------------------: | :--------------------: |
| Cora            | 87\.73                  |       **87\.93**        | 87\.91                 | 87\.83                 |
| Citeseer        | 77\.19                  | 77\.40                  | **77\.46**             | **77\.46**             |
| Pubmed          | 89\.73                  |       **89\.75**        | 89\.60                 | 89\.63                 |
| Chameleon       | 78\.14                  | 78\.27                  | 78\.36                 | **78\.53**             |
| Wisconsin       |       **88\.43**        | 87\.84                  | 88\.04                 | 88\.24                 |
| Texas           |       **87\.30**   |       **87\.30**        | 86\.76                 | 86\.76                 |
| Cornell         | 87\.03                  | 87\.84                  | 86\.76                 | **88\.11**             |
| Squirrel        | 73\.48                  | 74\.10                  | 73\.95                 | **74\.15**             |
| Actor           | 35\.67                  |       **35\.75**        | 35\.25                 | 35\.22                 |
| Actor(no-norm)  | 37\.63                  | **37\.75**              | 37\.67                 | 37\.62                 |

**To run node classification on different hops:**

**3-hop** : ```./run_classification_3_hop.sh```

**8-hop** : ```./run_classification_8_hop.sh```

**16-hop** : ```./run_classification_16_hop.sh```

**32-hop** : ```./run_classification_32_hop.sh```


In addition, we include model accuracy of Actor dataset without using hop-normalization, as model shows higher accuracy in this setting.

**Results with considering homophily/heterophily assumption in datasets**

| **Dataset** | **3-hop Accuracy(%)** | **8-hop Accuracy(%)** |
| :---------- | :-------------------: | :-------------------: |
| Cora        | 87\.61                | 88\.23                |
| Citeseer    | 77\.17                | 77\.35                |
| Pubmed      | 89\.70                | 89\.78                |
| Chameleon   | 78\.93                | 78\.95                |
| Wisconsin   | 88\.24                | 87\.65                |
| Texas       | 87\.57                | 87\.57                |
| Cornell     | 87\.30                | 87\.30                |
| Squirrel    | 73\.86                | 73\.94                |
| Actor       | 35\.38                | 35\.62                |

To get above results, please run

```./run_classification_feat_type.sh```


**Results with Sub Feature Setting**

Sub_Feature setting is when a subset of hop features is selected to train the model.

To get results with sum operation over hop features,

```./run_sub_feature_sum.sh```

To get results with concatenation operation over hop features,

```./run_sub_feature_cat.sh```

**Results with removing ReLU activation between the layers**

To get results with sum operation over hop features,

```./run_sub_feature_no_relu_sum.sh```

To get results with concatenation operation over hop features,

```./run_sub_feature_no_relu_cat.sh```

**ogbn-papers100M (large-scale dataset)**

Improved model than mentioned in the paper with extra FC layer.
Please run ```python process_large.py``` in folder named large_data first to create data splits. Pre-processed training splits (4-hops) are available for download [here](https://zenodo.org/record/5543949). Please download all pickle
files and save them into large_data folder.

Then run ```./run_ogbn_papers.sh``` to train the model.

Accuracy shown below is average over five runs with random seeds 0-4.

| **Dataset**     | **4-hop Accuracy (%)** |
| :-------------- | :---------------------: |
| ogbn-papers100M | 68\.07                 |



(Results may vary slightly with a different platform, e.g. use of different GPU. In such case, for best performance, some hyperparameter search may be required. Please refer to [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html) for more details.)

Datasets and parts of preprocessing code were taken from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn) and [GCNII](https://github.com/chennnM/GCNII) repositories. We thank the authors of these papers for sharing their code.


