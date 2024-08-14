import numpy as np
from .voxel_grid_3d import VoxelGrid3D
from .voxel_grid_2d import VoxelGrid2D


def increase_resolution(bins, resolution):
    max_val = [b.max() for b in bins]
    min_val = [b.min() for b in bins]
    pos_bins = [np.linspace(0, m, int(m//resolution)) for m in max_val]
    neg_bins = [np.linspace(n, -resolution, int(-n//resolution))
                for n in min_val]
    new_bins = [np.concatenate((n, p)) for n, p in zip(neg_bins, pos_bins)]
    return new_bins
