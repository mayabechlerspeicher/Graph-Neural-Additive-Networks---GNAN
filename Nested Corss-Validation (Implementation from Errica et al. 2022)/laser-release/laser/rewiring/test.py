import networkx as nx


def get_curvature_paper_graph():
    g = nx.empty_graph(7)

    edges = [
        [0, 1],
        [0, 3],
        [0, 2],
        [0, 4],
        [0, 6],
        [2, 5],
        [3, 5],
        [5, 1],
        [1, 6],
        [6, 4],
    ]

    for i, j in edges:
        g.add_edge(i, j)

    return g


def get_curvature_paper_graph_curvatures():
    # (edge): [d_i, d_j, triangles, squares_i, squares_j, degen_factor]
    curvature_components = {
        (0, 1): [5, 3, 1, 2, 1, 2],
        (0, 3): [5, 2, 0, 2, 1, 2],
        (0, 2): [5, 2, 0, 2, 1, 2],
        (0, 4): [5, 2, 1, 0, 0, 0],
        (0, 6): [5, 3, 2, 0, 0, 0],
        (2, 5): [2, 3, 0, 1, 2, 2],
        (3, 5): [2, 3, 0, 1, 2, 2],
        (5, 1): [3, 3, 0, 2, 1, 2],
        (1, 6): [3, 3, 1, 0, 0, 0],
        (6, 4): [3, 2, 1, 0, 0, 0],
    }

    Cs = {
        k: compute_curvature_from_components(*components)
        for k, components in curvature_components.items()
    }

    return Cs


def compute_curvature_from_components(
    d_i, d_j, triangles, squares_i, squares_j, degen_factor
):
    max_d = max(d_i, d_j)
    min_d = min(d_i, d_j)

    triangle_contributions = 2 * (triangles / max_d) + (triangles / min_d)
    square_contributions = (
        0
        if degen_factor == 0
        else (1 / (degen_factor * max_d)) * (squares_i + squares_j)
    )

    return 2 / d_i + 2 / d_j - 2 + triangle_contributions + square_contributions