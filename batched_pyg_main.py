import torch
import torch.utils.data as data
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


class MutagenicityDistanceDataset(data.Dataset):
    def __init__(self, root):
        """
        Loads the TUDataset 'Mutagenicity' from disk.
        """
        self.dataset = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
          x: [num_nodes, num_features], float
          dist_matrix: [num_nodes, num_nodes], float (BFS distances)
                       Off-diagonal = BFS distance, -1 if unreachable
          y: shape [1] or [num_classes]
        """
        pyg_data = self.dataset[idx]

        # Node features
        x = pyg_data.x.numpy()  # shape [N, in_features]

        # Graph label
        y = pyg_data.y.numpy()  # shape [1] in Mutagenicity (0 or 1)

        # Convert to networkx to compute BFS distances
        G = to_networkx(pyg_data, to_undirected=True)
        num_nodes = pyg_data.num_nodes

        dist_matrix = np.full((num_nodes, num_nodes), fill_value=-1, dtype=np.float32)
        for node in range(num_nodes):
            dist_matrix[node, node] = 0
            lengths = nx.shortest_path_length(G, source=node)
            for target_node, dist_val in lengths.items():
                dist_matrix[node, target_node] = dist_val

        # Convert to torch
        x_torch = torch.tensor(x, dtype=torch.float)
        dist_torch = torch.tensor(dist_matrix, dtype=torch.float)
        y_torch = torch.tensor(y, dtype=torch.long)  # for classification

        return x_torch, dist_torch, y_torch


def distance_collate_fn(batch):
    """
    batch: list of (x, dist, y) from __getitem__

    We'll produce:
      x_batch:       [sum_of_nodes_in_batch, in_features]
      dist_batch:    block-diagonal BFS distances (float32),
                     shape [sum_of_nodes_in_batch, sum_of_nodes_in_batch]
      y_batch:       [batch_size] (one label per graph)
      batch_vector:  [sum_of_nodes_in_batch], indicating graph index for each node
    """
    xs, dists, ys = zip(*batch)  # each is length = batch_size

    num_nodes_list = [x.shape[0] for x in xs]
    total_nodes = sum(num_nodes_list)
    batch_size = len(xs)

    # 1) Concatenate x
    x_batch = torch.cat(xs, dim=0)  # shape [total_nodes, in_features]

    # 2) Build block-diagonal dist_batch
    dist_batch = torch.full((total_nodes, total_nodes), fill_value=-1, dtype=torch.float)
    start_idx = 0
    for dist_mat in dists:
        n = dist_mat.shape[0]
        dist_batch[start_idx:start_idx + n, start_idx:start_idx + n] = dist_mat
        start_idx += n

    # 3) Build y_batch (graph-level labels)
    y_batch = torch.cat(ys, dim=0)  # shape [batch_size]

    # 4) Build a 'batch' vector: node -> graph index
    batch_vector = []
    for graph_idx, n_nodes in enumerate(num_nodes_list):
        batch_vector.append(torch.full((n_nodes,), fill_value=graph_idx, dtype=torch.long))
    batch_vector = torch.cat(batch_vector, dim=0)  # shape [total_nodes]

    return x_batch, dist_batch, y_batch, batch_vector


import torch
import torch.nn as nn


class TensorGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hidden_channels=16, device='cpu',
                 bias=True, dropout=0.0, is_graph_task=True):
        """
        Minimal GNAN-like model:
          - self.fs: list of MLPs (one per feature)
          - self.rho: MLP for distances
          - 'is_graph_task=True' => final output is [batch_size, out_channels]
        """
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.is_graph_task = is_graph_task

        # 1) Build self.fs: a separate MLP for each input feature
        self.fs = nn.ModuleList()
        for _ in range(in_channels):
            # simple 2-layer MLP
            layers = [
                nn.Linear(1, hidden_channels, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels, bias=bias)
            ]
            self.fs.append(nn.Sequential(*layers))

        # 2) Build self.rho: MLP for distance
        rho_layers = [
            nn.Linear(1, hidden_channels, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels, bias=bias)
        ]
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x_batch, dist_batch, batch_vector):
        """
        x_batch:      [N, in_channels]
        dist_batch:   [N, N], BFS distances, -1 for cross-graph
        batch_vector: [N], node -> graph index

        Returns:
          if self.is_graph_task: [num_graphs_in_batch, out_channels]
          else: [N, out_channels]
        """
        # 1) Feature transformation
        N, in_channels = x_batch.shape
        fx = torch.empty(N, in_channels, self.out_channels, device=self.device)
        for feat_idx in range(in_channels):
            feat_col = x_batch[:, feat_idx].view(-1, 1)  # [N,1]
            fx[:, feat_idx, :] = self.fs[feat_idx](feat_col)

        # fx shape: [N, in_channels, out_channels]

        # 2) Distance transformation
        # Flatten dist_batch -> shape [N*N, 1], run through self.rho
        dist_embed = self.rho(dist_batch.flatten().view(-1, 1))  # => [N*N, out_channels]
        dist_embed = dist_embed.view(N, N, self.out_channels)  # => [N, N, out_channels]

        # Zero out cross-graph pairs:
        mask = (dist_batch >= 0)  # True if not -1
        dist_embed[~mask] = 0.0

        # 3) Permute for batch matmul
        # m_dist: [C, N, N], fx_perm: [C, N, F]
        m_dist = dist_embed.permute(2, 0, 1)  # => [out_channels, N, N]
        fx_perm = fx.permute(2, 0, 1)  # => [out_channels, N, in_channels]

        # 4) Multiply => [out_channels, N, in_channels]
        mf = torch.matmul(m_dist, fx_perm)  # => [C, N, F]

        # 5) Sum over in_channels => [C, N]
        mf = mf.sum(dim=2)  # => [C, N]
        mf = mf.permute(1, 0)  # => [N, C]

        # 6) Aggregation
        if self.is_graph_task:
            # We have multiple graphs in the batch => scatter_add_ per graph
            num_graphs = batch_vector.max().item() + 1
            out_graph = torch.zeros(num_graphs, mf.size(1), device=mf.device)
            # batch_vector => shape [N]; expand to [N, C]
            graph_index = batch_vector.view(-1, 1).expand(-1, mf.size(1))
            out_graph.scatter_add_(0, graph_index, mf)
            return out_graph  # => [num_graphs_in_batch, out_channels]
        else:
            # node-level => just return [N, C]
            return mf



from torch.utils.data import DataLoader
import torch.optim as optim

# 1) Create dataset and loader
dataset = MutagenicityDistanceDataset(root='path_to_mutagenicity')
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=distance_collate_fn)

# 2) Create model
model = TensorGNAN(
    in_channels=dataset[0][0].shape[1],  # e.g. number of node features
    out_channels=8,
    n_layers=2,
    hidden_channels=16,
    is_graph_task=True
).to('cpu')

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()  # For binary classification with out_channels=8, you'd adapt accordingly

for epoch in range(100):
    total_loss = 0
    total_accuracy = 0
    for x_batch, dist_batch, y_batch, batch_vector in loader:
        x_batch = x_batch.to(model.device)
        dist_batch = dist_batch.to(model.device)
        y_batch = y_batch.to(model.device)
        batch_vector = batch_vector.to(model.device)

        optimizer.zero_grad()
        outputs = model(x_batch, dist_batch, batch_vector)  # => [batch_size, out_channels]

        # Suppose y_batch is shape [batch_size], with classes in {0,1}.
        # But our model has out_channels=8 => for a typical 2-class scenario, you'd want out_channels=2 or 1
        # We'll just illustrate:
        loss = loss_fn(outputs, y_batch)
        accuracy = (outputs.argmax(dim=-1) == y_batch).float().mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}, Accuracy: {total_accuracy / len(loader)}")


