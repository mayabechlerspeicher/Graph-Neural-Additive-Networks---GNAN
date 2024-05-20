# Implementation from https://github.com/kedar2/FoSR/blob/main/preprocessing/fosr.py
from torch_geometric.utils import to_networkx
from numba import jit, int64
import networkx as nx
import numpy as np
from math import inf
import torch
from dataclasses import dataclass
from ..transform import process_TUDataset

@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
    # chooses edge (u, v) to add which minimizes y[u]*y[v]
    n = x.size
    m = edge_index.shape[1]
    y = x / ((degrees + 1) ** 0.5)
    products = np.outer(y, y)
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        products[u, v] = inf
    for i in range(n):
        products[i, i] = inf
    smallest_product = np.argmin(products)
    return (smallest_product % n, smallest_product // n)


@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
    # returns array of degrees of all nodes
    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    degrees = np.zeros(num_nodes)
    m = edge_index.shape[1]
    for i in range(m):
        degrees[edge_index[0, i]] += 1
    return degrees


@jit(nopython=True)
def add_edge(edge_index, u, v):
    new_edge = np.array([[u, v], [v, u]])
    return np.concatenate((edge_index, new_edge), axis=1)


@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
    # given an edge_index, computes Ax, where A is the corresponding adjacency matrix
    n = x.size
    y = np.zeros(n)
    m = edge_index.shape[1]
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        y[u] += x[v]
    return y


@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    degrees = compute_degrees(edge_index, num_nodes=n)
    y = adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
    for i in range(n):
        if x[i] > 1e-9:
            return 1 - y[i] / x[i]
    return 0.0


@jit(nopython=True)
def _edge_rewire(
    edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50
):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(n) - 1
    degrees = compute_degrees(edge_index, num_nodes=n)
    for i in range(initial_power_iters):
        x = x - x.dot(degrees**0.5) * (degrees**0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
        x = y / np.linalg.norm(y)
    for I in range(num_iterations):
        i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
        edge_index = add_edge(edge_index, i, j)
        degrees[i] += 1
        degrees[j] += 1
        edge_type = np.append(edge_type, 1)
        edge_type = np.append(edge_type, 1)
        x = x - x.dot(degrees**0.5) * (degrees**0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
        x = y / np.linalg.norm(y)
    return edge_index, edge_type, x


def edge_rewire(
    edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5
):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(n) - 1
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    return _edge_rewire(
        edge_index,
        edge_type=edge_type,
        x=x,
        num_iterations=num_iterations,
        initial_power_iters=initial_power_iters,
    )

@dataclass
class FOSRTransform:
    num_snapshots: int = 1
    num_iterations: int = 50
    initial_power_iters: int = 5
    dataset: str = None

    def transform(self, g):
        g = process_TUDataset(g, self.dataset)

        edge_weights_rewired = torch.ones(g.edge_index.shape[1])

        try:
            new_edge_index, _, _ = edge_rewire(
                g.edge_index.numpy(), num_iterations=self.num_iterations
            )
            new_edge_index = torch.tensor(new_edge_index)
        except Exception as e:
            print(f"WARNING: Could not complete rewiring {e}")
            new_edge_index = torch.tensor([])

        g.edge_index = torch.cat((g.edge_index, new_edge_index), dim=1)

        if new_edge_index.shape[0] == 0:
            g.edge_weights = edge_weights_rewired
        elif self.num_snapshots == 1:
            new_edge_weights = torch.ones(new_edge_index.shape[1])
            g.edge_weights = torch.cat((edge_weights_rewired, new_edge_weights))
        elif self.num_snapshots == 2:
            new_edge_weights = torch.full((new_edge_index.shape[1],), 2)
            g.edge_weights = torch.cat((edge_weights_rewired, new_edge_weights))
        else:
            raise ValueError("FOSR requires 1 or 2 snapshots!")
        
        return g