import numpy as np

from algorithms.gnn_mappo.graph_builder import radius_graph


def test_radius_graph_without_self_loops() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    adjacency = radius_graph(positions, radius=2.0, self_loops=False)
    assert adjacency[0, 1] == 1.0
    assert adjacency[0, 2] == 0.0
    assert adjacency[0, 0] == 0.0
