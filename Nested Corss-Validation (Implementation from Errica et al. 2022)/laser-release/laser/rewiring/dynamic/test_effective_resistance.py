from .effective_resistance import compute_effective_resistance
from ..test import get_curvature_paper_graph
import itertools
import networkx as nx


def test_compute_effective_resistance():
    g = get_curvature_paper_graph()

    R = compute_effective_resistance(g)

    for (i, i_node), (j, j_node) in itertools.product(
        enumerate(g.nodes), enumerate(g.nodes)
    ):
        if i == j:
            continue

        fast_resistance = R[i, j]
        groundtruth_resistance = nx.resistance_distance(g, i_node, j_node)

        assert abs(fast_resistance - groundtruth_resistance) < 10e-5
        assert fast_resistance > 0
