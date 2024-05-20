from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

def dynamic_config(cfg):
    cfg.dynamic = CN()

    # Number of snapshots in the dynamic setting
    cfg.dynamic.num_snapshots = 1

    # Dynamic pre_transform to apply
    cfg.dynamic.pre_transform = "none"

    # The type of snapshot-wise aggregation to apply
    cfg.dynamic.aggregation = "sum"

    # Additions factor for dynamic rewiring
    cfg.dynamic.additions_factor = 0.5

    # Minimum additions for orbit-level rewiring
    cfg.dynamic.minimum_additions = 1

    # Rewiring method: either adjacency or resistance
    cfg.dynamic.rewiring_method = "adjacency"

    # Rewiring done either at the orbit (local) 
    # or graph (global) level
    cfg.dynamic.rewiring_level = "orbit"

    # Whether to shuffle or not on tie-breaks
    cfg.dynamic.shuffle = True

    """DIGL/FOSR/SDRF CONFIGS"""
    cfg.dynamic.num_iterations = 20

    # digl settings
    cfg.dynamic.teleport_proba = 0.05
    cfg.dynamic.sparsification_thresh = 1e-3
    

register_config("dynamic_cfg", dynamic_config)