import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.models.layer import LayerConfig

from torch_geometric.nn import RGCNConv, FastRGCNConv

from ..rewiring.transform import get_snapshot_edge_index, get_snapshot_edge_attr

class DynamicGeneralLayer(nn.Module):
    """
    General wrapper for layers

    Args:
        name (str): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, num_snapshots=None, **kwargs):
        super().__init__()
        self.num_snapshots = num_snapshots or cfg.dynamic.num_snapshots
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        # TODO CONSIDER num_snapshots==1 case and we don't have relation types
        # self.layer = RGCNConv(layer_config.dim_in, layer_config.dim_out, self.num_snapshots, root_weight=False)
        self.layers = nn.ModuleList([
            register.layer_dict[name](layer_config, **kwargs)
            for _ in range(self.num_snapshots)
        ])
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

        if cfg.dynamic.aggregation == "attention":
            # initialize attention weights with more mass on G_0
            activations = torch.zeros((self.num_snapshots, ))
            activations[0] = float(self.num_snapshots ** 2)
            self.attention_activations = nn.Parameter(activations)


    def forward(self, batch):
        # -1 is used on the edge_weights because the relations 
        # start at 0 when using RGCNConv, while the snapshots start at 1
        # edge_weights = (batch.edge_weights - 1).to(torch.long)
        # batch.x = self.layer(batch.x, batch.edge_index, edge_weights)
        xs = []
        edge_index_original = batch.edge_index
        
        if hasattr(batch, "edge_attr"):
            edge_attr_original = batch.edge_attr

        edge_attr_original
        for snapshot, layer in enumerate(self.layers):
            if self.num_snapshots == 1:
                edge_index_snapshot = batch.edge_index
            else:
                edge_index_snapshot = get_snapshot_edge_index(batch, snapshot_idx=snapshot+1)
            if edge_index_snapshot.shape[0] == 0:
                continue
            
            if self.num_snapshots > 1 and hasattr(batch, "edge_attr"):
                batch.edge_attr = get_snapshot_edge_attr(batch, snapshot_idx=snapshot+1)

            batch.edge_index = edge_index_snapshot.to(torch.long)
            xs.append(layer(batch).x)
            
            batch.edge_index = edge_index_original
            
            if hasattr(batch, "edge_attr"):
                batch.edge_attr = edge_attr_original

        batch.x = self._aggregate(xs, aggregation_type="sum")

        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)

        return batch

    def _aggregate(self, xs, aggregation_type=None):
        aggregation = cfg.dynamic.aggregation if aggregation_type is None else aggregation_type
        if aggregation == "attention":
            x = torch.stack(xs, dim=-1)
            attention_weights = F.softmax(self.attention_activations, dim=-1)
            x = attention_weights * x
            x = torch.sum(x, dim=-1)
        elif aggregation == "concat":
            x = torch.cat(xs, dim=-1)
        elif aggregation == "sum":
            xs = torch.stack(xs, dim=-1)
            x = torch.sum(xs, dim=-1)
        elif aggregation == "mean":
            xs = torch.stack(xs, dim=-1)
            x = torch.mean(xs, dim=-1)
        else:
            raise ValueError(f"{aggregation} not supported!")

        return x

def GNNLayer(dim_in, dim_out, num_snaphots=None, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return DynamicGeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg),
        num_snapshots=num_snaphots)


class DynamicGNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.num_snapshots = cfg.dynamic.num_snapshots

        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            num_snapshots = 1 if i < num_layers - 1 else self.num_snapshots
            layer = GNNLayer(d_in, dim_out, num_snaphots=num_snapshots)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif cfg.gnn.stage_type == 'skipconcat' and \
                    i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

class GNN(nn.Module):
    """
    General GNN model: encoder + stage + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        GNNStage = DynamicGNNStackStage
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

register_network("dynamic_gnn", GNN)