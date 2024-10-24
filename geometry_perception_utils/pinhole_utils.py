from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from multiprocessing.pool import ThreadPool
from geometry_perception_utils.vispy_utils.vispy_utils import plot_list_pcl
from scipy import interpolate
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous


class PinholeCamera:
    def __init__(self, cfg):
        self.shape = cfg.resolution
        self.fov = cfg.fov
        self.compute_default_grids()

    def compute_default_grids(self):
        h, w = self.shape
        u = np.linspace(0, w - 1, w).astype(int)
        v = np.linspace(0, h - 1, h).astype(int)
        uu, vv = np.meshgrid(u, v)
        self.default_pixel = np.vstack(
            (uu.flatten(), vv.flatten())).astype(np.int32)
        self.K = np.array(
            [
                [(w / 2.0) / np.tan(np.deg2rad(self.fov) / 2), 0, (w / 2.0)],
                [0, (h / 2.0) / np.tan(np.deg2rad(self.fov) / 2), (h / 2.0)],
                [0, 0, 1]
            ])  # (3, 3)

        # * Bearings vectors on the homogenous plane
        self.default_bearings_pp = uv2xyz(self.default_pixel, self.K)

        # * Bearing vectors on the unit sphere
        self.default_bearings_sph = self.default_bearings_pp / \
            np.linalg.norm(self.default_bearings_pp, axis=0, keepdims=True)

        # * Theta and phi angles range
        self.theta_range = np.linspace(
            -np.deg2rad(self.fov/2), np.deg2rad(self.fov/2) - self.fov/w, w)
        self.phi_range = np.linspace(
            -np.deg2rad(self.fov/2), np.deg2rad(self.fov/2) - self.fov/h, h)

    def get_color_pcl_from_depth_and_rgb_maps(self,
                                              color_map,
                                              depth_map,
                                              scaler=1):
        from geometry_perception_utils.image_utils import get_color_array

        color_pixels = get_color_array(color_map=color_map) / 255
        mask = depth_map.flatten() > 0
        pcl = (self.default_bearings_pp[:, mask] * scaler *
               get_color_array(color_map=depth_map)[0][mask])
        return pcl, color_pixels[:, mask]


def xyz2uv(xyz, K):
    assert xyz.shape[0] == 3
    __xyz = extend_array_to_homogeneous(xyz)
    return np.round(K[:2, :] @ __xyz - 0.5)


def uv2xyz(uv, K):
    assert uv.shape[0] == 2
    __uv = extend_array_to_homogeneous(uv + 0.5)
    return np.linalg.inv(K) @ __uv


def get_bearings_from_hfov(h, w, fov):
    fx, fy = np.deg2rad(fov), np.deg2rad(fov*h/w)
    x = np.linspace(-np.tan(fx/2), np.tan(fx/2), w)
    y = np.linspace(-np.tan(fy/2), np.tan(fy/2), h)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack((xx.flatten(), yy.flatten()))
    bearings = extend_array_to_homogeneous(xy)
    return bearings


def get_bearings_from_K(h, w, K):
    u = np.linspace(0, w - 1, w).astype(int)
    v = np.linspace(0, h - 1, h).astype(int)
    uu, vv = np.meshgrid(u, v)
    default_pixel = np.vstack(
        (uu.flatten(), vv.flatten())).astype(np.int32)

    # * Bearings vectors on the homogenous plane
    bearings_pp = uv2xyz(default_pixel, K)
    return bearings_pp


def project_pp_depth_from_K(depth_map, K, mask=None, epsilon=0.1, from_bearings=False):
    """
    Projects depth maps into 3D considering only the pixels in the mask
    """
    # bearing vectors
    h, w = depth_map.shape[:2]
    bearings = get_bearings_from_K(h, w, K)
    
    if mask is not None:
        m = mask.flatten() * depth_map.flatten() > epsilon
    else:
        m = depth_map.flatten() > epsilon
    
    if from_bearings:
        bearing_norm = bearings/np.linalg.norm(bearings, axis=0)
        xyz = depth_map.flatten()[m] * bearing_norm[:, m]
        return xyz, m
    xyz = depth_map.flatten()[m] * bearings[:, m]
    return xyz, m


def project_pp_depth_from_hfov(depth_map, hfov=90, mask=None, epsilon=0.1):
    """
    Projects depth maps into 3D considering only the pixels in the mask
    """

    # bearing vectors
    h, w = depth_map.shape[:2]  
    bearings = get_bearings_from_hfov(h, w, hfov)

    if mask is not None:
        m = mask.flatten() * depth_map.flatten() > epsilon
    else:
        m = depth_map.flatten() > epsilon
    xyz = depth_map.flatten()[m] * bearings[:, m]
    return xyz, m
