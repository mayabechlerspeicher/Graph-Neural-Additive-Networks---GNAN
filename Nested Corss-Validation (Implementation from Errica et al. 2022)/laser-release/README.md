# Locality-Aware Graph-Rewiring in GNNs

<img src="https://i.imgur.com/QUFyYPz.png" align="right" width="400" style="background-color:white;"/>

In this repository, we release the code required to replicate and expand upon our results in our work: `Locality-Aware Graph-Rewiring in GNNs`. We release our work under the MIT license.

The core of the functionality of LASER may be found in the `laser/rewiring` directory, with the details being described in the paper and in the docs within the code.

### Environment for LASER with Conda
We recommend building the environment through `Conda`. 
```bash
conda create -n laser python=3.9
conda activate laser

conda install pytorch=1.9 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.2 -c pyg -c conda-forge
conda install pandas scikit-learn

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

# Check https://www.dgl.ai/pages/start.html to install DGL based on your CUDA requirements
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html

pip install performer-pytorch
pip install torchmetrics==0.7.2
pip install ogb
pip install wandb
pip install torch_geometric=2.0.2 pytorch_lightning yacs torch_scatter torch_sparse numba pytest

conda clean --all
```


### Running LASER
The configuration files may be found in `/configs`, while utility running scripts may be found in `/run`. Here are some examples: 
```bash
conda activate laser

# Run automated tests
pytest . 

# Running LASER on PCQM-Contact
python main.py --cfg configs/Contact-laser/pcqm-contact-GCN-laserglobal.yaml wandb.use False

# Running the LASER ablation study on Peptides-func
bash run/run_peptides_laser_ablation_func.sh

# Running the SAGE results on Peptides
bash run/run_sage_lrgb.sh

# Running the LASER TUDatasets results (warning that there is some stochasticity in the results)
bash run/run_tudataset_baselines.sh
bash run/run_tudataset_laser.sh
```
We note that for the TUDatasets experiments, as discussed in the paper, there is a lot of stochasticity which makes reproducing the results less reliable. This is instead not the case for the LRGB experiments.

### Acknowledgments
In order to accurately replicate part of the baseline results, our code builds upon the [Long-Range Graph Benchmark (LRGB) repository](https://github.com/vijaydwivedi75/lrgb). We additionally borrow the implementations of [First Order Spectral Rewiring (FOSR)](https://github.com/kedar2/FoSR) and [Stochastic Discrete Ricci Flow (SDRF)](https://github.com/jctops/understanding-oversquashing) from their respective authors.
