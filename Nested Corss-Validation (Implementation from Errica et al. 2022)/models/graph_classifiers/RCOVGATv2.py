import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import Optional
import torch.nn as nn
from typing import Tuple, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)


class RCOVGATv2Layer(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels)

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
    ):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l = None
        x_r = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr,
                index: Tensor, ptr,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr.relu()

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class RCOVGATv2Model(nn.Module):
    def __init__(self, dim_features, dim_target, config, linear=False,
                 bias=False, aggr='sum', pool='mean', dropout=0.0, batch_norm=False,
                 residual=False,
                 init_std=1e-4,
                 graphless=False, graph_cls=True, graph_level_task=True, edge_dim=1):
        super(RCOVGATv2Model, self).__init__()

        self.out_channels = dim_target
        self.hidden_channels = config['dim_embedding']
        self.num_layers = config['num_layers']

        self.linear = linear
        self.bias = bias

        self.aggr = aggr
        self.pool = pool

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        self.init_std = init_std
        self.graphless = graphless
        self.graph_level_task = graph_level_task

        self.graph_convs = []
        self.lns = []
        self.graph_convs.append(
            RCOVGATv2Layer(in_channels=dim_features, out_channels=self.hidden_channels, edge_dim=edge_dim, bias=bias))
        self.lns.append(pyg.nn.norm.LayerNorm(self.hidden_channels, affine=False))
        for _ in range(1, self.num_layers):
            self.graph_convs.append(
                RCOVGATv2Layer(in_channels=self.hidden_channels, out_channels=self.hidden_channels, edge_dim=edge_dim,
                               bias=bias))
            self.lns.append(pyg.nn.norm.LayerNorm(self.hidden_channels, affine=False))
        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_cls: self.pool = global_mean_pool
        self.readout = nn.Linear(self.hidden_channels, dim_target, bias=bias)
        self.activation = nn.ReLU() if not linear else nn.Identity()

        if init_std > 0:
            self.init_params()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param, mean=0, std=self.init_std)
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        if 'edge_weight' in inputs:
            edge_weight = inputs.edge_weight.view(-1, 1)
        else:
            edge_weight = torch.ones(edge_index.shape[1], 1).to(edge_index.device)
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index, edge_attr=edge_weight)
            h = self.activation(h)
            if not l == self.num_layers - 1:
                h = self.lns[l](x=h, batch=batch)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.graph_level_task:  # case node cls
            h = self.pool(h, batch)

        y = self.readout(h)

        return y
