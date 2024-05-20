import torch.nn.functional as F
import torch_geometric.transforms as T
from dataclasses import dataclass

def get_snapshot_edge_index(data, snapshot_idx):
    """
    Helper method to access the rewirings of the class. As a speedup,
    the method also saves to the data object the snapshot rewirings. This is to
    avoid recomputing the edge_index mask on every iteration which is wasteful.
    This means that subsequent accesses to the rewirings are O(1).

    Args:
        data: PyTorch Geometric data object.
        snapshot_idx: Snasphot we want the edge_index for.
    """
    if snapshot_idx == 0:
        raise ValueError("Snapshots start at 1.")

    if not hasattr(data, "edge_rewirings_store"):
        data.edge_rewirings_store = {}

    if snapshot_idx in data.edge_rewirings_store:
        return data.edge_rewirings_store[snapshot_idx]

    # if we could not find the rewiring in storage,
    # calculate it and store it
    rewiring = data.edge_index.T[data.edge_weights == snapshot_idx].T
    data.edge_rewirings_store[snapshot_idx] = rewiring

    # assert rewiring.shape[1] > 0, f"Rewiring for {snapshot_idx} not found."

    return rewiring

def get_snapshot_edge_attr(data, snapshot_idx):
    if snapshot_idx == 0:
        raise ValueError("Snapshots start at 1.")

    if not hasattr(data, "edge_attr_rewirings_store"):
        data.edge_attr_rewirings_store = {}

    if snapshot_idx in data.edge_attr_rewirings_store:
        return data.edge_attr_rewirings_store[snapshot_idx]

    # if we could not find the rewiring in storage,
    # calculate it and store it
    attr = data.edge_attr[data.edge_weights == snapshot_idx, :]
    data.edge_attr_rewirings_store[snapshot_idx] = attr

    # assert rewiring.shape[1] > 0, f"Rewiring for {snapshot_idx} not found."

    return attr

def one_hot_transform(g):
    g.y = F.one_hot(g.y, num_classes=6)
    return g

def process_TUDataset(g, dataset):
    if dataset in ["MUTAG", "ENZYMES", "PROTEINS"]:
        g = one_hot_transform(g)
    elif dataset in ["REDDIT-BINARY", "IMDB-BINARY", "COLLAB"]:
        g = one_hot_transform(g)
        g = T.Constant()(g)
    
    return g

@dataclass
class TUDatasetTransform:
    dataset: str = None

    def transform(self, g):
        return process_TUDataset(g,dataset=self.dataset)