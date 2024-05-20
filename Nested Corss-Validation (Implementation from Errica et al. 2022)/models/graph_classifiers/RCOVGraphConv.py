from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from typing import Optional
import torch.nn as nn
from typing import Tuple, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)


class RCOVGraphConvLayer(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: str = 'add',
            bias: bool = True,
            edge_dim: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)
        if edge_dim is not None:
            self.lin = Linear(edge_dim, in_channels[0])
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if self.lin is not None:
            edge_weight = self.lin(edge_weight)
        return x_j if edge_weight is None else x_j + edge_weight.relu()

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)


class RCOVGraphConvmodel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, edge_dim=1, **kwargs):
        super(RCOVGraphConvmodel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.graph_convs = []
        self.graph_convs.append(
            RCOVGraphConvLayer(in_channels=in_channels, out_channels=hidden_channels, edge_dim=edge_dim, bias=bias))
        for _ in range(1, num_layers):
            self.graph_convs.append(
                RCOVGraphConvLayer(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=edge_dim,
                                   bias=bias))
        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

        self.init_params()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch, edge_weight = inputs.x, inputs.edge_index, inputs.batch, inputs.edge_weight.view(-1, 1)
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index, edge_weight=edge_weight)
            h = self.activation(h)

        h = self.pool(h, batch)
        out = self.readout(h)
        return out
