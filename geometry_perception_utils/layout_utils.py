import numpy as np
from geometry_perception_utils.spherical_utils import xyz2uv
from geometry_perception_utils.hohonet_utils.vis_utils import layout_2_depth

def clip_boundary(boundary, threshold):
    dist = np.linalg.norm(boundary, axis=0)
    mask = dist > threshold
    if np.sum(mask) > 0:
        boundary[:, mask] = threshold * boundary[:, mask] / dist[mask] 
    return boundary

def ly2depth_from_list_corners(list_corners, camera_height, ceiling_height, shape=(512, 1024)):
    ceiling_cor = [p + np.array((0, -ceiling_height, 0)).reshape(3, 1) for p in list_corners]
    floor_cor = [p + np.array((0, camera_height, 0)).reshape(3, 1) for p in list_corners]
    
    corners = []
    [(corners.append(ceiling_cor[i]), corners.append(floor_cor[i]) ) for i in range(ceiling_cor.__len__())]
    uv_corners = xyz2uv(np.hstack(corners)).T
    
    depth = layout_2_depth(uv_corners, shape[0], shape[1])
    return depth    