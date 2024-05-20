import pytest
from .fosr import FOSRTransform
from torch_geometric.datasets.zinc import ZINC
import torch
from ..transform import get_snapshot_edge_index

class TestTransform:
    def setup_class(self):
        self.dataset = ZINC(root="datasets-test/ZINC/", subset=True, split="test")
    
    def test_snapshot_implementation(self):
        test_index = 10
        num_iterations = 50

        one_snapshot = FOSRTransform(num_snapshots=1, num_iterations=num_iterations)
        two_snapshot = FOSRTransform(num_snapshots=2, num_iterations=num_iterations)

        sample_data = self.dataset[test_index]

        sample_one = one_snapshot.transform(sample_data.clone())
        sample_two = two_snapshot.transform(sample_data.clone())

        assert torch.equal(sample_one.x, sample_two.x)
        assert sample_one.edge_index.shape[1] == sample_two.edge_index.shape[1]
        assert sample_one.edge_weights[0] == 1 and sample_one.edge_weights[-1] == 1
        assert sample_two.edge_weights[0] == 1 and sample_two.edge_weights[-1] == 2