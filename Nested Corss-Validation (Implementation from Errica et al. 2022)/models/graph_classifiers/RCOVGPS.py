from torch_geometric.nn import global_add_pool, global_mean_pool

from torch_geometric.nn import GraphConv, GATConv, GINConv, GCNConv, ChebConv, GATv2Conv, GINEConv, CGConv, NNConv
import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
import torch.nn as nn


class RCOV_GPSLayer(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            act: str = 'relu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            attn_type: str = 'multihead',
            attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')


class RCOV_GPSmodel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, linear=False,
                 bias=False, aggr='sum', pool='mean', dropout=0.0, batch_norm=False,
                 residual=False,
                 init_std=1e-4,
                 graphless=False, graph_cls=True, graph_level_task=True):
        super(RCOV_GPSmodel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

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

        assert not batch_norm, "Batch norm not implemented yet"
        assert not residual, "Residual connections not implemented yet"

        self.graph_convs = []
        self.lns = []
        conv_layer = GraphConv(in_channels=in_channels, out_channels=hidden_channels, bias=bias)
        self.graph_convs.append(GPSConv(channels=in_channels, conv=conv_layer))
        self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        for _ in range(1, num_layers):
            conv_layer = GraphConv(in_channels=in_channels, out_channels=hidden_channels, bias=bias)
            self.graph_convs.append(RCOV_GPSLayer(channels=in_channels, conv=conv_layer))
            self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_cls: self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
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
        x, edge_index, batch, edge = inputs.x, inputs.edge_index, inputs.batch, inputs.edge_attr.view(-1, 1)
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index, edge_weight=edge)
            h = self.activation(h)
            if not l == self.num_layers - 1:
                h = self.lns[l](x=h, batch=batch)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.graph_level_task:  # case node cls
            h = self.pool(h, batch)

        y = self.readout(h)

        return y
