import math
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from multiprocessing.pool import ThreadPool
from geometry_perception_utils.vispy_utils.vispy_utils import plot_list_pcl

class SphericalCamera:
    def __init__(self, shape):
        self.shape = shape
        self.compute_default_grids()

    def compute_default_grids(self):
        h, w = self.shape
        u = np.linspace(0, w - 1, w).astype(int) + 0.5
        v = np.linspace(0, h - 1, h).astype(int) + 0.5
        uu, vv = np.meshgrid(u, v)
        self.default_pixel = np.vstack((uu.flatten(), vv.flatten())).astype(np.int)
        self.default_bearings = uv2xyz(self.default_pixel, self.shape)
        r = np.pi / w
        self.theta_range = np.linspace(-np.pi+r, np.pi-r, w)

    def xyz2phi_coords(self, xyz, xyz_type="floor"):
        assert xyz_type in ["floor", "ceiling"], "xyz_type must be either floor or ceiling"
        assert xyz.shape[0] == 3, "xyz must be a 3xN array"    
        theta_coords, phi_coords= xyz2sph(xyz)
        
        r = 0.5 * np.pi / self.shape[1]
        def map_phi_coords(theta):
            idx = np.where(abs(theta_coords - theta) < r)[0]
            if xyz_type == "floor":
                phi = phi_coords[idx].max()
            if xyz_type == "ceiling":
                phi = phi_coords[idx].min()
            return theta, phi   
        
        # pool = ThreadPool(4)
        sph_coords =[
            # pool.apply_async(map_phi_coords, (theta,))
            map_phi_coords(theta)
            for theta in  self.theta_range
        ]
        # sph_coords = [thread.get() for thread in list_threads]
            
        return np.vstack(sph_coords).T[1, :]
          
    def phi_coords2xyz(self, phi_coords):
        assert phi_coords.size == 1024, "phi_coords must be a 1024 array"
        
        x = np.cos(phi_coords) * np.sin(self.theta_range)
        y = np.sin(phi_coords)
        z = np.cos(phi_coords) * np.cos(self.theta_range)

        return np.vstack((x, y, z))
    
    def xyz2uv(self, pcl):
        pass
    
    def get_color_pcl_from_depth_and_rgb_maps(self, color_map, depth_map, scaler=1):
        from geometry_perception_utils.image_utils import get_color_array
        
        color_pixels = get_color_array(color_map=color_map) / 255
        mask = depth_map.flatten() > 0
        pcl = (
            self.default_bearings[:, mask]
            * scaler
            * get_color_array(color_map=depth_map)[0][mask]
        )
        return pcl, color_pixels[:, mask]

# ! ok
def uv2xyz(uv, shape):
    """
    Projects uv vectors to xyz vectors (bearing vector)
    """
    sph = uv2sph(uv, shape)
    theta = sph[0]
    phi = sph[1]

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.vstack((x, y, z))

# ! ok 
def uv2sph(uv, shape):
    """
    Projects a set of uv points into spherical coordinates (theta, phi)
    """
    H, W = shape
    theta = 2 * np.pi * ((uv[0]) / W - 0.5)
    phi = np.pi * ((uv[1]) / H - 0.5)
    return np.vstack((theta, phi))


def sph2xyz(sph):
    """
    Projects spherical coordinates (theta, phi) to euclidean space xyz
    """
    theta = sph[:, 0]
    phi = sph[:, 1]

    x = math.cos(phi) * math.sin(theta)
    y = math.sin(phi)
    z = math.cos(phi) * math.cos(theta)

    return np.vstack((x, y, z))


def sphere_normalization(xyz):
    norm = np.linalg.norm(xyz, axis=0)
    return xyz / norm


def xyz2sph(xyz):
    xyz_n = xyz / np.linalg.norm(xyz, axis=0, keepdims=True)

    normXZ = np.linalg.norm(xyz[(0, 2), :], axis=0, keepdims=True)

    phi_coords = np.arcsin(xyz_n[1, :])
    theta_coord = np.sign(xyz[0, :]) * np.arccos(xyz[2, :] / normXZ)
    return theta_coord.ravel(), phi_coords.ravel()

    
def xyz2horizon_depth(xyz):
    """
    Returns the horizon depth of a set of xyz coordinates
    """
    return np.linalg.norm(xyz[(0, 2), :], axis=0, keepdims=True)


def xyz2uv(xyz, shape=(512, 1024)):
    """
    Projects XYZ array into uv coord
    """
    theta_coord, phi_coord = xyz2sph(xyz)

    u = np.clip(np.floor((0.5 * theta_coord / np.pi + 0.5) * shape[1]), 0, shape[1] - 1)
    v = np.clip(np.floor((phi_coord / np.pi + 0.5) * shape[0]), 0, shape[0] - 1)
    return np.vstack((u, v)).astype(int)

