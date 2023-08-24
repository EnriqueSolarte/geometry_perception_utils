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
        u = np.linspace(0, w - 1, w).astype(int) + 0.5
        v = np.linspace(0, h - 1, h).astype(int) + 0.5
        uu, vv = np.meshgrid(u, v)
        self.default_pixel = np.vstack(
            (uu.flatten(), vv.flatten())).astype(np.int32)
        self.K = np.array(
            [
                [(w / 2.0) / np.tan(np.deg2rad(self.fov) / 2), 0, (w / 2.0)],
                [0, (h / 2.0) / np.tan(np.deg2rad(self.fov) / 2), (h / 2.0)],
                [0, 0, 1]
            ])  # (3, 3)
        self.default_bearings_pp = uv2xyz(self.default_pixel, self.K)
        self.default_bearings_sph = self.default_bearings_pp / np.linalg.norm(self.default_bearings_pp, axis=0, keepdims=True)        
        self.theta_range = np.linspace(
            -np.deg2rad(self.fov/2), np.deg2rad(self.fov/2) - self.fov/w, w)
        self.phi_range = np.linspace(
            -np.deg2rad(self.fov/2), np.deg2rad(self.fov/2) - self.fov/h, h)

    def xyz2bearings(self, xyz):
        raise NotImplementedError

    def bearings2xyz(self, bearings):
        raise NotImplementedError
        
    def bearing2uv(self, bearings):
        raise NotImplementedError
        
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



def uv2xyz(uv, K):
    assert uv.shape[0] == 2    
    __uv = extend_array_to_homogeneous(uv)
    return np.linalg.inv(K) @ __uv
