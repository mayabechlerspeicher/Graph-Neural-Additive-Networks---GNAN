# GNAN - An Interpretable and Effective Graph Neural Additive Network
![](fig.png)
Now, you can have interpretability and accuracy at the same time when learning on graphs! 🤩 🤩 🤩 
GNAN is the first interpretable-by-design Graph Neural Network, delivering a white-box approach that matches the accuracy of top-performing black-box GNNs. By leveraging Generalized Additive Models and adapting them to graph data, GNAN allows you to visualize exactly what your model learns—offering transparent explanations for the model and its predictions, more effective debugging, and more informed model selection based on learned priors rather than guesswork. Paper and code in the first comment.

This repository contains the implementation of the Graph Neural Additive Networks (GNAN) as described in the paper [The Intelligible and Effective Graph Neural Additive Networks](https://arxiv.org/pdf/2406.01317).
Graph Neural Additive Networks (GNAN) are designed to be fully interpretable, providing both global and local explanations at the feature and graph levels through direct visualization of the model.


### Requirements
* Python 3.9
* PyTorch 1.0.0
* PyTorch Geometric 1.0.0
* scikit-learn 0.20.0
* numpy 1.15.4
* scipy 1.1.0
* pandas 0.23.4

# Run
To run an experiment, set the parameters in the run.sh file , then run bash run.sh
