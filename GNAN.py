class TensorGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', limited_m=True, normalize_m=True, is_graph_task=False,
                 final_agg='sum'):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.limited_m = limited_m
        self.normalize_m = normalize_m
        self.fs = nn.ModuleList()
        self.is_graph_task = is_graph_task

        for _ in range(in_channels):
            if num_layers == 1:
                curr_f = [nn.Linear(1, self.actual_output_dim_f, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, num_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, self.actual_output_dim_f, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

        m_bias = True
        if is_graph_task:  m_bias = False
        if num_layers == 1:
            self.m = [nn.Linear(1, self.out_channels, bias=m_bias)]

        else:
            self.m = [nn.Linear(1, hidden_channels, bias=m_bias), nn.ReLU()]
            for _ in range(1, num_layers - 1):
                self.m.append(nn.Linear(hidden_channels, hidden_channels, bias=m_bias))
                self.m.append(nn.ReLU())
            self.m.append(nn.Linear(hidden_channels, self.out_channels, bias=m_bias))
        self.m = nn.Sequential(*self.m)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
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
        if self.normalize_m:
            node_distances = torch.div(node_distances, inputs.normalization_matrix)
        m_dist = self.m(node_distances.flatten().view(-1, 1)).view(x.size(0), x.size(0), self.out_channels)
        m_dist_perm = torch.permute(m_dist, (2, 0, 1))

        mf = torch.matmul(m_dist_perm, fx_perm)

        if not self.is_graph_task:
            out = torch.sum(mf, dim=2)
        else:
            hidden = torch.sum(mf, dim=1)
            out = torch.sum(hidden, dim=1).view(1, -1)
        return out.T


class GNAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', limited_m=True, normalize_m=True, m_per_feature=False):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.limited_m = limited_m
        self.normalize_m = normalize_m
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
        if m_per_feature:
            self.ms = nn.ModuleList()
            for _ in range(in_channels):
                if num_layers == 1:
                    self.m = [nn.Linear(1, out_channels, bias=bias)]
                else:
                    self.m = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                    for _ in range(1, num_layers - 1):
                        self.m.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                        self.m.append(nn.ReLU())
                    if limited_m:
                        self.m.append(nn.Linear(hidden_channels, 1, bias=bias))
                    else:
                        self.m.append(nn.Linear(hidden_channels, out_channels, bias=bias))

                self.ms.append(nn.Sequential(*self.m))
        else:
            if num_layers == 1:
                self.m = [nn.Linear(1, out_channels, bias=bias)]
            else:
                self.m = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                for _ in range(1, num_layers - 1):
                    self.m.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    self.m.append(nn.ReLU())
                if limited_m:
                    self.m.append(nn.Linear(hidden_channels, 1, bias=bias))
                else:
                    self.m.append(nn.Linear(hidden_channels, out_channels, bias=bias))

            self.m = nn.Sequential(*self.m)

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
            m_dist = self.m(node_dists.view(-1, 1))
            if self.normalize_m:
                if m_dist.size(1) == 1:
                    m_dist = torch.div(m_dist, normalization.view(-1, 1))
                else:
                    for i in range(m_dist.size(1)):  # iterate number of classes
                        m_dist[:, i] = torch.div(m_dist[:, i], normalization)
            pred_for_node = torch.sum(torch.mul(m_dist, f_sums), dim=0)
            stacked_results[j] = pred_for_node.view(1, -1)

        return stacked_results

    def print_m_params(self):
        for name, param in self.m.named_parameters():
            print(name, param)
