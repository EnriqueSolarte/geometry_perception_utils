
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from geometry_perception_utils.spherical_utils import xyz2sph, sph2xyz

def generate_pcl_by_roi_theta_phi(theta, phi, n_pts, min_d, max_d):
    """
    It generates a PCL of n_pts points in 3D give a range of theta and phi
    (spherical coordinates)
    """
    assert min_d < max_d
    assert n_pts > 0
    assert len(theta) == len(phi) == 2
    assert theta[0] < theta[1]
    assert phi[0] < phi[1]

    random_theta = np.random.uniform(np.radians(theta[0]),
                                     np.radians(theta[1]), n_pts)
    random_phi = np.random.uniform(np.radians(phi[0]), np.radians(phi[1]),
                                   n_pts)
    random_d = np.random.uniform(min_d, max_d, n_pts)
    pcl = np.array([
        np.cos(random_phi) * np.sin(random_theta), -np.sin(random_phi),
        np.cos(random_phi) * np.cos(random_theta)
    ])
    pcl *= random_d
    return pcl


def add_spherical_noise_to_pcl(pcl, std=0.1):
    assert pcl.shape[0] in (3, 4)
    
    theta, phi = xyz2sph(pcl[:3, :])

    theta += np.random.normal(0, std, pcl.shape[1])
    phi += np.random.normal(0, std, pcl.shape[1])

    return sph2xyz(np.vstack((theta, phi)))
    

def add_outliers_to_pcl(pcl, inliers=1):
    """
    It randomly select vector into the pcl array and redefines its values
    """
    assert pcl.shape[0] in (3, 4)
    assert inliers <= pcl.shape[1]
    outliers_src = np.random.randint(0, pcl.shape[1], pcl.shape[1] - int(inliers))
    outliers_des = np.random.randint(0, pcl.shape[1], pcl.shape[1] - int(inliers))

    pcl[:, outliers_src] = -pcl[:, outliers_des]

    return pcl
