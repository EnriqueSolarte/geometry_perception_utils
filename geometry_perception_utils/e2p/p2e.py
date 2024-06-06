import numpy as np
from geometry_perception_utils.spherical_utils import xyz2uv, sph2xyz
from geometry_perception_utils.vispy_utils import plot_list_pcl
from geometry_perception_utils.geometry_utils import get_rot_matrix_from_two_vectors


def pinhole_mask_on_equirectangular(u_deg=0, v_deg=0, out_hw=(512, 1024), model='bilinear'):

    # * at least as the max of the output image size
    pinhole_resolution = max(out_hw)  # resolution of the pinhole camera

    x = np.linspace(-0.5, 0.5, pinhole_resolution)
    y = np.linspace(-0.5, 0.5, pinhole_resolution)
    xx, yy = np.meshgrid(x, y)
    xyz = np.vstack((xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())))

    orig_vector = np.array([0, 0, 1])
    sph_coord = np.array([0, np.deg2rad(v_deg)]).reshape(2, 1)
    dst_vector = sph2xyz(sph_coord)
    if v_deg != 0:
        rot = get_rot_matrix_from_two_vectors(
            orig_vector.ravel(), dst_vector.ravel())
    else:
        rot = np.eye(3)

    xyz = rot @ xyz

    uv = xyz2uv(xyz, out_hw)

    equi_img = np.zeros((out_hw[0], out_hw[1]))
    equi_img[uv[1, :], uv[0, :]] = 1
    equi_img = np.roll(equi_img, axis=1, shift=int(
        np.round(u_deg / 360 * out_hw[1])))
    return equi_img
