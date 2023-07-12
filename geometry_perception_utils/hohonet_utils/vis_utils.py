import numpy as np
from geometry_perception_utils.hohonet_utils.utils import cor_2_1d

def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth