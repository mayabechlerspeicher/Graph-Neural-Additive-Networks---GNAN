from torch_geometric.graphgym.config import cfg

from ..rewiring.dynamic.laser_global import LaserGlobalTransform
from ..rewiring.dynamic.laser_random import LaserRandomTransform
from ..rewiring.sdrf.sdrf import SDRFTransform
from ..rewiring.fosr.fosr import FOSRTransform
from ..rewiring.digl.digl import DIGLTransform

def get_dynamic_pre_transform(dataset):
    config_pre_transform = cfg.dynamic.pre_transform
    num_snapshots = cfg.dynamic.num_snapshots
    additions_factor = cfg.dynamic.additions_factor
    # exclusive = cfg.dynamic.exclusive
    minimum_additions = cfg.dynamic.minimum_additions
    rewiring_method = cfg.dynamic.rewiring_method
    rewiring_level = cfg.dynamic.rewiring_level
    shuffle = cfg.dynamic.shuffle
    model = cfg.model.type

    # digl
    teleport_proba = cfg.dynamic.teleport_proba
    sparsification_thresh = cfg.dynamic.sparsification_thresh
    
    num_iterations = cfg.dynamic.num_iterations

    if model not in ["dynamic_gnn"]:
        return None

    if config_pre_transform == "none":
        return None
    
    if config_pre_transform == "laserglobal":
        return LaserGlobalTransform(
            num_snapshots=num_snapshots,
            additions_factor=additions_factor,
            minimum_additions=minimum_additions,
            rewiring_level=rewiring_level,
            shuffle=shuffle,
            dataset=dataset
        ).transform

    if config_pre_transform == "laserrandom":
        return LaserRandomTransform(
            num_snapshots=num_snapshots,
            additions_factor=additions_factor,
            minimum_additions=minimum_additions,
            rewiring_level=rewiring_level,
            dataset=dataset
        ).transform

    if config_pre_transform == "fosr":
        return FOSRTransform(
            num_snapshots=num_snapshots,
            num_iterations=num_iterations,
            dataset=dataset
        ).transform
    
    if config_pre_transform == "sdrf":
        return SDRFTransform(
            num_snapshots=num_snapshots,
            num_iterations=num_iterations,
            dataset=dataset
        ).transform
    

    if config_pre_transform == "digl":
        return DIGLTransform(
            alpha=teleport_proba, eps=sparsification_thresh, dataset=dataset
        ).transform

    raise Exception(f"{config_pre_transform} not supported as a pre_transform.")