import torch_geometric
import networkx as nx
import numpy as np
import torch
import math
from .effective_resistance import compute_effective_resistance
from numba import njit
from dataclasses import dataclass
from ..transform import process_TUDataset

@dataclass
class LaserRandomTransform:
    """
    This class is the same as the LaserGlobalTransform class, 
    with the difference being that we sample the edges uniformly instead
    of being guided by the connectivity measure (powers of adj for example).

    Args:
        num_snapshots: The number of snapshots to produce.
        additions_factor: The number of edges to sample from each orbit.
        minimum_additions: The minimum number of edges to sample.
        rewiring_level: `orbit` sampling supported only.
        dataset: The name of the dataset.
    """
    num_snapshots: int = 5
    additions_factor: float = 1
    minimum_additions: int = 1
    rewiring_level: str = "orbit"
    dataset: str = None
    
    def transform(self, g):
        """
        The transform method. Takes a graph and applies the required transform.

        Args:
            g: Graph.
        """
        g = process_TUDataset(g, self.dataset)

        g_nx = torch_geometric.utils.to_networkx(g, to_undirected=True)

        edge_index, edge_weights = self._create_rewirings(g_nx)

        g.edge_index, g.edge_weights = edge_index, edge_weights

        return g

    def _create_rewirings(self, g):
        """
        Creates the rewiring method following the LASER procedure. The edges 
        are sampled uniformly at random.

        Args:
            g: Graph.
        """
        # as everything is symmetric we only consider the lower half
        D = np.tril(nx.floyd_warshall_numpy(g))

        edge_index_rewired = torch_geometric.utils.from_networkx(g).edge_index
        edge_weights_rewired = torch.ones(edge_index_rewired.shape[1])

        for r in range(2, self.num_snapshots + 1):
            new_edge_index = self._get_new_edge_index(D, r)

            added_size = 0 if len(new_edge_index.shape) == 1 else new_edge_index.shape[1]
            new_edge_weights = torch.full((added_size,), r)

            edge_index_rewired = torch.cat((edge_index_rewired, new_edge_index), dim=1)
            edge_weights_rewired = torch.cat((edge_weights_rewired, new_edge_weights))

        return edge_index_rewired.to(torch.long), edge_weights_rewired

    def _get_new_edge_index(self, D, r):
        """
        Helper method to get the new edge index.

        Args:
            D: Matrix of distances.
            r: Radius we are considering.
        """
        if self.rewiring_level == "orbit":
            return self._edge_additions_for_snapshot_orbit_level(D, r)            

        raise ValueError(f"Only `orbit` level rewirings supported!")

    def _edge_additions_for_snapshot_orbit_level(self, D, r):
        """
        Sample edges uniformly for each orbit.

        Args:
            D: Matrix of distances.
            r: Radius we are considering.
        """
        n = D.shape[0]

        added_edges = []
        for i in range(n):
            D_row = D[i,:]
            D_row_mask = D_row == r
            D_row_mask[i] = False

            D_row_masked = D_row[D_row_mask]
            orbit_size = D_row_masked.shape[0]
            add_per_node = max(self.minimum_additions, math.ceil(orbit_size * self.additions_factor))

            best_k_idxs = self._get_k_random_idxs(D_row_masked, add_per_node) 

            D_mask_indices = np.where(D_row_mask)
            selected = np.array(D_mask_indices)[0][best_k_idxs]

            for j in selected:
                added_edges.append([i, j])
                added_edges.append([j, i])

        return torch.tensor(added_edges).T


    def _get_k_random_idxs(self, D_masked, k):
        """
        Sample k indices at random from D_masked.

        Args:
            D_masked: The indices to sample from.
            k: Number of indices to sample.
        """
        size = D_masked.shape[0]
        
        if size <= k:
            random_k_idxs = np.arange(size)
        else:
            random_k_idxs = np.random.choice(size, k)

        return random_k_idxs