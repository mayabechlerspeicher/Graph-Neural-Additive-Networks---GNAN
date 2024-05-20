import networkx as nx
import numpy as np
from numba import njit, cuda

def compute_effective_resistance(g, approximate=False):
    """
    Compute the effective resistance matrix for a graph g.
    This is a dense matrix with entry ij being the effective
    resistance between i and j in g.

    Args:
        g: Graph.
    """
    n_nodes = g.number_of_nodes()
    L = nx.laplacian_matrix(g, list(g)).toarray().astype(np.float64)

    R = _compute_effective_resistance(L, n_nodes, approximate)

    return R


@njit
def _compute_effective_resistance(L, n_nodes, approximate):
    """
    Jitted implementation.

    Compute the effective resistance matrix for a graph g.
    This is a dense matrix with entry ij being the effective
    resistance between i and j in g.

    Args:
        L: Laplacian matrix.
        n_nodes: Number of nodes in the graph.
    """
    rcond = 1e-5 if approximate else 1e-10
    L_inv = np.linalg.pinv(L, rcond=rcond)
    L_inv_diag = np.diag(L_inv)
    L_inv_aa = np.broadcast_to(L_inv_diag, (n_nodes, n_nodes))
    R = L_inv_aa - 2 * L_inv + L_inv_aa.T

    return R

def effective_resistance_edge_addition(g, num_additions, approximate):
    """
    Adds num_additions of edges to g based on the effective resistance
    R of the graph. We iteratively compute R and connect nodes
    with the highest effective resistance.

    Args:
        g: Graph.
        num_additions: Number of edges to add to g.
        approximate: Approximate the algorithm by not recomputing the 
            effective resistance matrix for a number of APPROX_ITERATIONS.
    """
    node_list = list(g.nodes)
    APPROX_ITERATIONS = int(len(node_list) * 0.2)
    approx_counter = APPROX_ITERATIONS
    added_edges = []

    for _ in range(num_additions):
        if approximate:
            if approx_counter == APPROX_ITERATIONS:
                approx_counter = 0
                R = compute_effective_resistance(g, approximate)
            else:
                approx_counter += 1
        else:
            R = compute_effective_resistance(g, approximate)

        max_resistance_index = np.unravel_index(R.argmax(), R.shape)
        g.add_edge(
            node_list[max_resistance_index[0]], node_list[max_resistance_index[1]]
        )
        added_edges.append([max_resistance_index[0], max_resistance_index[1]])
        added_edges.append([max_resistance_index[1], max_resistance_index[0]])

        # edge has been added now so removing it for approximate ER 
        R[max_resistance_index] = 0 

    return g, added_edges
