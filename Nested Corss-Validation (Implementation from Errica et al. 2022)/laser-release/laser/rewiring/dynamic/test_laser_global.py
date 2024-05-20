from .laser_global import LaserGlobalTransform
from ..transform import get_snapshot_edge_index 
import torch
from torch_geometric.datasets.zinc import ZINC
import pytest
import numpy as np

class TestTransform:
    def setup_class(self):
        self.dataset = ZINC(root="datasets-test/ZINC/", subset=True, split="test")

    @pytest.mark.parametrize('test_index', [10, 15])
    @pytest.mark.parametrize('num_snapshots', [1,2,4])
    @pytest.mark.parametrize('rewiring_level', ["orbit", "graph"])
    def test_transform_multiple_snapshots(self, test_index, num_snapshots, rewiring_level):
        er_transform = LaserGlobalTransform(num_snapshots=num_snapshots, rewiring_level=rewiring_level)

        sample_data = self.dataset[test_index]
        transformed = er_transform.transform(sample_data.clone())

        rewired_edge_index = get_snapshot_edge_index(transformed, 1)

        assert torch.equal(rewired_edge_index, sample_data.edge_index)

    def test_get_best_k_idxs(self):
        er_transform = LaserGlobalTransform()
        
        M_masked = np.array([514, 10.14, 1245, 1234, 10.15])
        k = 3
        # check stochasticity by trying this many times
        for _ in range(100):
            best_k = er_transform._get_best_k_idxs(M_masked, k)

            # with adjacency rewiring we pick the k lowest
            to_find = set((1, 4, 0))


            for element in best_k:
                assert element in to_find
                to_find.remove(element)
            
            assert len(to_find) == 0
    
    def test_get_best_k_idxs_tie_breaks(self):
        er_transform = LaserGlobalTransform()
        
        element_counter = {0:0, 1:0, 2:0, 3:0}
        M_masked = np.array([1.01, 1, 1, 0.99])
        k = 2
        runs = 100000
        tolerance = 0.05
        # here we check the distribution of the elements given tie breaks
        for _ in range(runs):
            best_k = er_transform._get_best_k_idxs(M_masked, k)

            for element in best_k:
                element_counter[element] += 1

        assert element_counter[3] == runs
        assert abs(element_counter[1] / element_counter[2] - 1) < tolerance

            
    @pytest.mark.parametrize('test_index', [10, 15])
    @pytest.mark.parametrize('rewiring_level', ["orbit", "graph"])
    def test_transform_multiple_snapshots(self, test_index, rewiring_level):
        num_snapshots = 5
        er_transform = LaserGlobalTransform(num_snapshots=num_snapshots, rewiring_level=rewiring_level)

        sample_data = self.dataset[test_index]
        transformed = er_transform.transform(sample_data.clone())

        found_edges = set()
        for k in range(2, num_snapshots + 1):
            rewired_edge_index = get_snapshot_edge_index(transformed, k)
            for u, v in rewired_edge_index.T:
                assert (u, v) not in found_edges
                found_edges.add((u,v))

            assert rewired_edge_index.shape[0] > 0