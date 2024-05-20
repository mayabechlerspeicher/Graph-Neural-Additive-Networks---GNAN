from .laser_random import LaserRandomTransform
from .laser_global import LaserGlobalTransform
from ..transform import get_snapshot_edge_index
import torch
from torch_geometric.datasets.zinc import ZINC
import pytest
import numpy as np

class TestTransform:
    def setup_class(self):
        self.dataset = ZINC(root="datasets-test/ZINC/", subset=True, split="test")

    def test_get_random_k_idxs(self):
        transform = LaserRandomTransform()

        D_masked = np.array([1, 2, 3, 4])
        k = 3

        element_counter = {0: 0, 1:0, 2:0, 3:0}
        runs = 100000
        tolerance = 0.05

        for _ in range(runs):
            chosen_k = transform._get_k_random_idxs(D_masked, k)
            
            for element in chosen_k:
                element_counter[element] += 1
        
        for _, counter in element_counter.items():
            assert abs(counter / runs - k / D_masked.shape[0]) < tolerance
        
    def test_density(self):
        transform_random = LaserRandomTransform(num_snapshots=2)
        transform_global = LaserGlobalTransform(num_snapshots=2)

        for i in range(10):
            data = self.dataset[i]
            assert transform_global.transform(data.clone()).edge_index.shape == transform_random.transform(data.clone()).edge_index.shape