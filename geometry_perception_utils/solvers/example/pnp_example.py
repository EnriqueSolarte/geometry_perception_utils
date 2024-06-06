from geometry_perception_utils.pcl_utils import add_spherical_noise_to_pcl
from geometry_perception_utils.geometry_utils import (
    eulerAnglesToRotationMatrix, extend_array_to_homogeneous,
    evaluate_error_in_transformation)
from geometry_perception_utils.spherical_utils import sphere_normalization
from geometry_perception_utils.solvers.pnp import PnP
from geometry_perception_utils.vispy_utils import plot_list_pcl
import time
import numpy as np
import cv2


def get_random_transform():
    cam_pose = np.eye(4)
    euler_angles = np.random.uniform(-np.pi, np.pi, 3)
    R = eulerAnglesToRotationMatrix(euler_angles)
    t = np.random.uniform(-10, 10, 3)
    cam_pose[0:3, 0:3] = R
    cam_pose[0:3, 3] = t
    return cam_pose


def main():

    pnp = PnP()
    timing = []
    rot_errors = []
    t_errors = []
    d_errors = []
    for _ in range(1000000):

        pcl_wc = np.random.uniform(-1, 1, (3, 1000))

        cam_c2w = get_random_transform()

        pcl_cc = cam_c2w[:3, :] @ extend_array_to_homogeneous(pcl_wc)

        bearings_cc = sphere_normalization(pcl_cc)
        bearings_ccn = add_spherical_noise_to_pcl(
            bearings_cc.copy(), std=2*np.pi/1024)

        tic_toc = time.time()
        cam_pnp = pnp.recover_pose(landmarks=pcl_wc.copy(),
                                   bearings=bearings_ccn.copy())

        timing.append(time.time() - tic_toc)

        rot_e, t_e, d_e = evaluate_error_in_transformation(transform_gt=cam_c2w,
                                                           transform_est=cam_pnp)

        rot_errors.append(rot_e)
        t_errors.append(t_e)
        d_errors.append(d_e)
        print(cam_pnp[:3, 3])
        print(cam_c2w[:3, 3])

        print(
            f"Mean Rot-error (deg) : {np.mean(rot_errors)} - {rot_errors.__len__()}")
        print(
            f"Mean t-error (deg) : {np.mean(t_errors)} - {t_errors.__len__()}")
        print(
            f"Mean t-error (dist-e) : {np.mean(d_errors)} - {t_errors.__len__()}")

        print("time: {}".format(np.mean(timing)))
        print(
            "====================================================================="
        )


if __name__ == '__main__':
    main()
