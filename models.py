import torch
from torch_geometric.nn import GraphConv, GINConv, GATv2Conv, GraphSAGE, TransformerConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F

class GATv2Model(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, graph_level_task=False):
        super(GATv2Model, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.graph_convs.append(GATv2Conv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        for l in range(1, num_layers):
            self.graph_convs.append(GATv2Conv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, graph_level_task=False):
        super(GraphSAGEModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.graph_convs.append(GraphSAGE(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        for l in range(1, num_layers):
            self.graph_convs.append(GraphSAGE(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GraphConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 graph_level_task=False):
        super(GraphConvModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.lns = []
        self.graph_convs.append(GraphConv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        for l in range(1, num_layers):
            self.graph_convs.append(GraphConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))
            self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_level_task: self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)
            h = self.lns[l](x=h, batch=batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GraphTranformerModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, graph_level_task=False):
        super(GraphSAGEModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.graph_convs.append(TransformerConv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        for l in range(1, num_layers):
            self.graph_convs.append(TransformerConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GINModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 graph_level_task=False):
        super(GINModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.lns = []
        nn_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias), nn.ReLU())
        self.graph_convs.append(GINConv(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        bias=bias, nn=nn_layer))
        self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        for l in range(1, num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU())
            self.graph_convs.append(GINConv(in_channels=hidden_channels,
                                            out_channels=hidden_channels,
                                            bias=bias, nn=nn_layer))
            self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_level_task: self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)
            h = self.lns[l](x=h, batch=batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class NAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.fs = nn.ModuleList()

        for _ in range(in_channels):
            if num_layers == 1:
                curr_f = [nn.Linear(1, out_channels, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, num_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index]
            feature_col = feature_col.view(-1, 1)
            feature_col = self.fs[feature_index](feature_col)
            fx[:, feature_index] = feature_col

        f_sums = fx.sum(dim=1)
        return f_sums


class TensorGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', rho_per_feature=False, normalize_rho=True, is_graph_task=False, readout_n_layers=1):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.rho_per_feature = rho_per_feature
        self.normalize_rho = normalize_rho
        self.fs = nn.ModuleList()
        self.is_graph_task = is_graph_task
        self.readout_n_layers = readout_n_layers

        self.actual_output_dim_f = 1 if is_graph_task and readout_n_layers > 0 else out_channels
        self.actual_output_dim_rho = 1 if not rho_per_feature or (is_graph_task and readout_n_layers > 0) else out_channels

        for _ in range(in_channels):
            if n_layers == 1:
                curr_f = [nn.Linear(1, self.actual_output_dim_f, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, n_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, self.actual_output_dim_f, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

        rho_bias = True
        if is_graph_task:  rho_bias = False
        if n_layers == 1:
            self.rho = [nn.Linear(1, self.actual_output_dim_rho, bias=rho_bias)]

        else:
            self.rho = [nn.Linear(1, hidden_channels, bias=rho_bias), nn.ReLU()]
            for _ in range(1, n_layers - 1):
                self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=rho_bias))
                self.rho.append(nn.ReLU())
            self.rho.append(nn.Linear(hidden_channels, self.actual_output_dim_rho, bias=rho_bias))
        self.rho = nn.Sequential(*self.rho)

        if is_graph_task and self.readout_n_layers > 0:
            self.readout_nam = NAM(in_channels, out_channels, readout_n_layers, hidden_channels, bias, dropout,
                                   device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, node_distances = inputs.x, inputs.edge_index, inputs.node_distances
        fx = torch.empty(x.size(0), x.size(1), self.actual_output_dim_f).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index]
            feature_col = feature_col.view(-1, 1)
            feature_col = self.fs[feature_index](feature_col)
            fx[:, feature_index] = feature_col

        fx_perm = torch.permute(fx, (2, 0, 1))
        if self.normalize_rho:
            node_distances = torch.div(node_distances, inputs.normalization_matrix)
        m_dist = self.rho(node_distances.flatten().view(-1, 1)).view(x.size(0), x.size(0), self.actual_output_dim_rho)
        m_dist_perm = torch.permute(m_dist, (2, 0, 1))

        mf = torch.matmul(m_dist_perm, fx_perm)

        if not self.is_graph_task:
            out = torch.sum(mf, dim=2)

        else:
            hidden = torch.sum(mf, dim=1)
            if self.readout_n_layers > 0:
                out = self.readout_nam(hidden)
            else:
                out = torch.sum(hidden, dim=1).view(1, -1)
        return out.T


class GNAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', normalize_rho=True, rho_per_feature=False):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.rho_per_feature = rho_per_feature
        self.normalize_rho = normalize_rho
        self.fs = nn.ModuleList()

        for _ in range(in_channels):
            if num_layers == 1:
                curr_f = [nn.Linear(1, out_channels, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, num_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))
        if rho_per_feature:
            self.rhos = nn.ModuleList()
            for _ in range(in_channels):
                if num_layers == 1:
                    self.rho = [nn.Linear(1, out_channels, bias=bias)]
                else:
                    self.rho = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                    for _ in range(1, num_layers - 1):
                        self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                        self.rho.append(nn.ReLU())
                    if not rho_per_feature:
                        self.rho.append(nn.Linear(hidden_channels, 1, bias=bias))
                    else:
                        self.rho.append(nn.Linear(hidden_channels, out_channels, bias=bias))

                self.rhos.append(nn.Sequential(*self.rho))
        else:
            if num_layers == 1:
                self.rho = [nn.Linear(1, out_channels, bias=bias)]
            else:
                self.rho = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                for _ in range(1, num_layers - 1):
                    self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    self.rho.append(nn.ReLU())
                if not rho_per_feature:
                    self.rho.append(nn.Linear(hidden_channels, 1, bias=bias))
                else:
                    self.rho.append(nn.Linear(hidden_channels, out_channels, bias=bias))

        self.rho = nn.Sequential(*self.rho)

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs, node_ids=None):
        x, edge_index, node_distances = inputs.x, inputs.edge_index, inputs.node_distances
        if node_ids is None:
            node_ids = range(x.size(0))
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index]
            feature_col = feature_col.view(-1, 1)
            feature_col = self.fs[feature_index](feature_col)
            fx[:, feature_index] = feature_col

        f_sums = fx.sum(dim=1)
        stacked_results = torch.empty(len(node_ids), self.out_channels).to(self.device)
        for j, node in enumerate(node_ids):
            node_dists = node_distances[node]
            normalization = inputs.normalization_matrix[node]
            rho_dist = self.rho(node_dists.view(-1, 1))
            if self.normalize_rho:
                if rho_dist.size(1) == 1:
                    rho_dist = torch.div(rho_dist, normalization.view(-1, 1))
                else:
                    for i in range(rho_dist.size(1)):  # iterate number of classes
                        rho_dist[:, i] = torch.div(rho_dist[:, i], normalization)
            pred_for_node = torch.sum(torch.mul(rho_dist, f_sums), dim=0)
            stacked_results[j] = pred_for_node.view(1, -1)

        return stacked_results

    def print_rho_params(self):
        for name, param in self.rho.named_parameters():
            print(name, param)
