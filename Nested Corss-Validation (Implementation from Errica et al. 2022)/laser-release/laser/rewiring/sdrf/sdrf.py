# SDRF preprocessing, from https://github.com/jctops/understanding-oversquashing

from numba import jit, prange
import numpy as np
import torch
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from dataclasses import dataclass
from ..transform import process_TUDataset

def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0:
                C[i, j] = 0
                break

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                break

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C


@jit(nopython=True)
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    for I in prange(dim_i):
        for J in prange(dim_j):
            i = i_neighbors[I]
            j = j_neighbors[J]

            if (i == j) or (A[i, j] != 0):
                D[I, J] = -1000
                break

            # Difference in degree terms
            if j == x:
                d_in_x += 1
            elif i == y:
                d_out_y += 1

            if d_in_x * d_out_y == 0:
                D[I, J] = 0
                break

            if d_in_x > d_out_y:
                d_max = d_in_x
                d_min = d_out_y
            else:
                d_max = d_out_y
                d_min = d_in_x

            # Difference in triangles term
            A2_x_y = A2[x, y]
            if (x == i) and (A[j, y] != 0):
                A2_x_y += A[j, y]
            elif (y == j) and (A[x, i] != 0):
                A2_x_y += A[x, i]

            # Difference in four-cycles term
            sharp_ij = 0
            lambda_ij = 0
            for z in range(N):
                A_z_y = A[z, y] + 0
                A_x_z = A[x, z] + 0
                A2_z_y = A2[z, y] + 0
                A2_x_z = A2[x, z] + 0

                if (z == i) and (y == j):
                    A_z_y += 1
                if (x == i) and (z == j):
                    A_x_z += 1
                if (z == i) and (A[j, y] != 0):
                    A2_z_y += A[j, y]
                if (x == i) and (A[j, z] != 0):
                    A2_x_z += A[j, z]
                if (y == j) and (A[z, i] != 0):
                    A2_z_y += A[z, i]
                if (z == j) and (A[x, i] != 0):
                    A2_x_z += A[x, i]

                TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            D[I, J] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
            )
            if lambda_ij > 0:
                D[I, J] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = np.zeros((len(i_neighbors), len(j_neighbors)))

    _balanced_forman_post_delta(
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf(
    data, loops=10, remove_edges=True, removal_bound=0.5, tau=1, is_undirected=False
):
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    m = data.edge_index.shape[1]

    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type
    if is_undirected:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    else:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = 1.0
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    C = np.zeros((N, N))

    for x in range(loops):
        can_add = True
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin()
        x = ix_min // N
        y = ix_min % N
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))
        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for i, j in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            edge_type = np.append(edge_type, 1)
            edge_type = np.append(edge_type, 1)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break
    return from_networkx(G).edge_index, torch.tensor(edge_type)


@dataclass
class SDRFTransform:
    num_snapshots: int = 1
    num_iterations: int = 50
    dataset: str = None

    def transform(self, g):
        g = process_TUDataset(g, self.dataset)

        edge_weights_rewired = torch.ones(g.edge_index.shape[1])

        new_edge_index, _ = sdrf(
            g, loops=self.num_iterations, remove_edges=False, is_undirected=True
        )
        new_edge_index = torch.tensor(new_edge_index)

        g.edge_index = torch.cat((g.edge_index, new_edge_index), dim=1).to(torch.long)

        if self.num_snapshots == 1:
            new_edge_weights = torch.ones(new_edge_index.shape[1])
            g.edge_weights = torch.cat((edge_weights_rewired, new_edge_weights))
        elif self.num_snapshots == 2:
            new_edge_weights = torch.full((new_edge_index.shape[1],), 2)
            g.edge_weights = torch.cat((edge_weights_rewired, new_edge_weights))
        else:
            raise ValueError("SDRF requires 1 or 2 snapshots!")

        return g