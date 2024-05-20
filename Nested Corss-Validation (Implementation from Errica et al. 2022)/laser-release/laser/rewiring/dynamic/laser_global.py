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
class LaserGlobalTransform:
    """
    This class implements the LASER method from the paper. It implements
    the `transform` method that takes a graph and outputs its LASER rewiring
    given the parameters fed to the constructor. 

    Args:
        num_snapshots: The number of snapshots to produce.
        additions_factor: The number of edges to sample from each orbit.
        minimum_additions: The minimum number of edges to sample.
        rewiring_level: `orbit` sampling supported only.
        shuffle: True iff we want to uniformly sample from tie-breaks.
        dataset: The name of the dataset.
    """
    num_snapshots: int = 5
    additions_factor: float = 1
    minimum_additions: int = 1
    rewiring_level: str = "orbit"
    shuffle: bool = True
    dataset: str = None
    fast: bool = True
    
    def transform(self, g):
        """
        The transform method. Takes a graph and applies the required transform.

        Args:
            g: Graph.
        """
        g = process_TUDataset(g, self.dataset)

        g_nx = torch_geometric.utils.to_networkx(g, to_undirected=True)

        edge_index, edge_weights = self._create_rewirings(g_nx)

        if hasattr(g, "edge_attr"):
            to_add = edge_weights.shape[0] - g.edge_attr.shape[0]
            # assert to_add >= 0

            if to_add > 0:
                virtual_attrs = torch.full((to_add, ), 0.1).unsqueeze(-1)
                # print(g.edge_attr.shape, virtual_attrs.shape)
                edge_attr = torch.cat((g.edge_attr, virtual_attrs),  dim=0)

                g.edge_attr = edge_attr

                # # TODO can use GCN style weights here if we want to normalize
                # g.edge_attr = torch.full(edge_weights.shape[0], 0.1).unsqueeze(-1)

        g.edge_index, g.edge_weights = edge_index, edge_weights
        
        return g

    def _create_rewirings(self, g):
        """
        Creates the rewiring method following the LASER procedure.

        Args:
            g: Graph.
        """
        A = nx.adjacency_matrix(g).toarray() + np.eye(g.number_of_nodes())
        A[A == 2] = 1 # in case the matrix already had a diagonal

        # we don't need to compute floyd warshall as we just care about 
        # the neighbourhoods up to self.num_snapshots 
        if self.fast:
            D = A
            A_curr = A 
            A_next = A @ A

            for r in range(2, self.num_snapshots + 1):
                # clip matrices to 1 and 0
                A_next[A_next != 0] = 1 
                A_curr[A_curr != 0] = 1

                A_difference = A_next - A_curr
                D_difference = r * A_difference
                D += D_difference
                
                A_curr = A_next
                A_next = A_next @ A
        
            D = np.tril(D)
            
        else:
            # as everything is symmetric we only consider the lower half
            D = np.tril(nx.floyd_warshall_numpy(g))

        # choosing powers of 2 for faster relative computation
        M = np.linalg.matrix_power(A, 8)

        edge_index_rewired = torch_geometric.utils.from_networkx(g).edge_index
        edge_weights_rewired = torch.ones(edge_index_rewired.shape[1])

        for r in range(2, self.num_snapshots + 1):
            new_edge_index = self._get_new_edge_index(D, M, r)

            added_size = 0 if len(new_edge_index.shape) == 1 else new_edge_index.shape[1]
            new_edge_weights = torch.full((added_size,), r)

            edge_index_rewired = torch.cat((edge_index_rewired, new_edge_index), dim=1)
            edge_weights_rewired = torch.cat((edge_weights_rewired, new_edge_weights))

        return edge_index_rewired.to(torch.long), edge_weights_rewired

    def _get_new_edge_index(self, D, M, r):
        """
        Helper method to get the new edge index. In practice, we only 
        support the `orbit`-level rewiring that preserves the number of 
        additions per orbit. We have also implemented a graph level 
        operation, but found that this does not perform as well.

        Args:
            D: Matrix of distances.
            M: Matrix of 'connectivity measures' (powers of adj for example)
            r: Radius we are considering.
        """
        if self.rewiring_level == "orbit":
            return self._edge_additions_for_snapshot_orbit_level(D, M, r)            
        elif self.rewiring_level == "graph":
            return self._edge_additions_for_snapshot_graph_level(D, M, r)

        raise ValueError(f"Only `orbit` or `graph` level rewirings supported!")

    def _edge_additions_for_snapshot_graph_level(self, D, M, r):
        """
        LASER rewiring done at graph level. 
        Note: This is not used in the experiments in the paper.

        Args:
            D: Matrix of distances.
            M: Matrix of 'connectivity measures' (powers of adj for example)
            r: Radius we are considering.
        """
        D_flat, M_flat = D.flatten(), M.flatten()

        mask = D_flat == r
        M_masked = M_flat[mask]

        add_total = max(1, math.ceil(M_masked.shape[0] * self.additions_factor))

        best_k_idxs = self._get_best_k_idxs(M_masked, add_total) 

        D_mask_indices = np.where(mask)[0]

        selected = np.array(D_mask_indices)[best_k_idxs]
        selected = np.unravel_index(selected, M.shape)

        added_edges = []
        for u, v in zip(*selected):
            added_edges.append([u, v])
            added_edges.append([v, u])

        return torch.tensor(added_edges).T

    def _edge_additions_for_snapshot_orbit_level(self, D, M, r):
        """
        LASER rewiring done at the orbit level. This is the procedure
        we have found to work best and it is what is described in the 
        paper.

        Args:
            D: Matrix of distances.
            M: Matrix of 'connectivity measures' (powers of adj for example)
            r: Radius we are considering.
        """
        n = D.shape[0]

        added_edges = []
        for i in range(n):
            D_row, M_row = D[i,:], M[i,:]
            D_row_mask = D_row == r
            D_row_mask[i] = False

            M_row_masked = M_row[D_row_mask]
            orbit_size = M_row_masked.shape[0]
            add_per_node = max(self.minimum_additions, round(orbit_size * self.additions_factor))

            best_k_idxs = self._get_best_k_idxs(M_row_masked, add_per_node)

            D_mask_indices = np.where(D_row_mask)
            selected = np.array(D_mask_indices)[0][best_k_idxs]

            for j in selected:
                added_edges.append([i, j])
                # added_edges.append([j, i])

        return torch.tensor(added_edges).T
    
    def _get_best_k_idxs(self, M_masked, k):
        """
        Sample k indices given the connectivity measure. We sample
        from the first k_idxs. If `self.shuffle`, then we add very 
        small noise in such a way to break tie-breaks uniformly.

        Args:
            M_masked: The connectivity measures.
            k: Number of indices to sample.
        """
        size = M_masked.shape[0]
        
        # To break tie breaks, we add very small IID noise
        # such that it randomly shuffles only elements with the 
        # same value (effectively this allows us to efficiently
        # do random sampling)
        if self.shuffle:
            eps = 10e-8
            M_masked = M_masked + np.random.normal(0, eps, size)

        if size <= k:
            lowest_k_idxs = np.arange(size)
        else:
            lowest_k_idxs = np.argpartition(M_masked, k)[:k]

        return lowest_k_idxs