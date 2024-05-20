from typing import Callable, Optional, Union
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.nn.inits import reset
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class RCOVGINLayer(MessagePassing):

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, edge_index,
                edge_attr=None, size=None):

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'RCOVGINLayer'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr.relu()).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class RCOVGINModel(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config, linear=False,
                 bias=False, init_std=1e-4,
                 graphless=False, graph_cls=True, graph_level_task=True, edge_dim=1, **kwargs):
        super(RCOVGINModel, self).__init__()

        self.config = config
        self.hidden_channels = config['dim_embedding']
        self.num_layers = config['num_layers']

        self.out_channels = dim_target

        self.linear = linear
        self.bias = bias

        self.init_std = init_std
        self.graphless = graphless
        self.graph_level_task = graph_level_task

        self.graph_convs = []
        nn_layer = nn.Sequential(nn.Linear(dim_features, self.hidden_channels, bias=bias), nn.ReLU())
        self.graph_convs.append(RCOVGINLayer(nn=nn_layer, edge_dim=edge_dim))
        for _ in range(1, self.num_layers):
            cur_layer = nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels, bias=bias))
            self.graph_convs.append(RCOVGINLayer(nn=cur_layer, edge_dim=edge_dim))
        self.graph_convs = nn.ModuleList(self.graph_convs)
        if graph_cls: self.pool = global_mean_pool
        self.readout = nn.Linear(self.hidden_channels, dim_target, bias=bias)
        self.activation = nn.ReLU() if not linear else torch.nn.Identity()

        if init_std > 0:
            self.init_params()
        if graphless:
            self.set_graphless()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def set_graphless(self):
        for name, param in self.named_parameters():
            if 'lin_l' in name:
                param.requires_grad = False
                param.fill_(0)
                print('turned off parameter: ', name)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        if 'edge_weight' in inputs:
            edge_att = inputs.edge_weight.view(-1, 1)
        else:
            edge_att = torch.ones(edge_index.shape[1], 1).to(edge_index.device)
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index, edge_attr=edge_att)
            h = self.activation(h)

        if self.graph_level_task:  # case node cls
            h = self.pool(h, batch)

        y = self.readout(h)

        return y
