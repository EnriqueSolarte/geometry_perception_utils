import math

import numpy as np
from pyquaternion import Quaternion


def get_quaternion_from_matrix(matrix):
    """
    Returns the [qx, qy, qz qw] quaternion vector for the passed matrix (SE3 or SO3)
    """
    q = Quaternion(matrix=matrix)
    return (
        np.array((q.x, q.y, q.z, q.w)) if q.w > 0 else -
        np.array((q.x, q.y, q.z, q.w))
    )


def tum_pose2matrix44(l, seq="xyzw"):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    if seq == "wxyz":
        if q[0] < 0:
            q *= -1
        q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    else:
        if q[3] < 0:
            q *= -1
        q = Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3],
        )
    transform = np.eye(4)
    transform[0:3, 0:3] = q.rotation_matrix
    transform[0:3, 3] = np.array(t)

    return transform


def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def get_xyz_from_phi_coords(phi_coords):
    """
    Computes the xyz PCL from the ly_data (bearings_phi / phi_coords)
    """
    bearings_floor = get_bearings_from_phi_coords(phi_coords=phi_coords[1, :])

    # ! Projecting bearing to 3D as pcl --> boundary
    # > Forcing ly-scale = 1
    ly_scale = 1 / bearings_floor[1, :]
    pcl_floor = ly_scale * bearings_floor
    return pcl_floor


def get_bearings_from_phi_coords(phi_coords):
    """
    Returns 3D bearing vectors (on the unite sphere) from phi_coords
    """
    W = phi_coords.__len__()
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    bearings_y = -np.sin(phi_coords)
    bearings_x = np.cos(phi_coords) * np.sin(theta_coords)
    bearings_z = np.cos(phi_coords) * np.cos(theta_coords)
    return np.vstack((bearings_x, bearings_y, bearings_z))


def stack_camera_poses(list_poses):
    """
    Stack a list of camera poses using Kronecker product
    https://en.wikipedia.org/wiki/Kronecker_product
    """
    M = np.zeros((list_poses.__len__() * 3, list_poses.__len__() * 4))
    for idx in range(list_poses.__len__()):
        aux = np.zeros((list_poses.__len__(), list_poses.__len__()))
        aux[idx, idx] = 1
        M += np.kron(aux, list_poses[idx][0:3, :])
    return M


def extend_array_to_homogeneous(array):
    """
    Returns the homogeneous form of a vector by attaching
    a unit vector as additional dimensions
    Parameters
    ----------
    array of (3, n) or (2, n)
    Returns (4, n) or (3, n)
    -------
    """
    try:
        assert array.shape[0] in (2, 3, 4)
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples))))

    except:
        assert array.shape[1] in (2, 3, 4)
        array = array.T
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples)))).T


def extend_vector_to_homogeneous_transf(vector):
    """
    Creates a homogeneous transformation (4, 4) given a vector R3
    :param vector: vector R3 (3, 1) or (4, 1)
    :return: Homogeneous transformation (4, 4)
    """
    T = np.eye(4)
    if vector.__class__.__name__ == "dict":
        T[0, 3] = vector["x"]
        T[1, 3] = vector["y"]
        T[2, 3] = vector["z"]
    elif type(vector) == np.array:
        T[0:3, 3] = vector[0:3, 0]
    else:
        T[0:3, 3] = vector[0:3]
    return T


def eulerAnglesToRotationMatrix(angles):
    theta = np.zeros((3))

    if angles.__class__.__name__ == "dict":
        theta[0] = angles["x"]
        theta[1] = angles["y"]
        theta[2] = angles["z"]
    else:
        theta[0] = angles[0]
        theta[1] = angles[1]
        theta[2] = angles[2]

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    # R = np.dot(R_z, np.dot(R_y, R_x))
    R = R_x @ R_y @ R_z
    return R


def rotationMatrixToEulerAngles(R):
    """rotationMatrixToEulerAngles retuns the euler angles of a SO3 matrix"""
    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def vector2skew_matrix(vector):
    """
    Converts a vector [3,] into a matrix [3, 3] for cross product operations. v x v' = [v]v' where [v] is a skew representation of v
    :param vector: [3,]
    :return: skew matrix [3, 3]
    """
    vector = vector.ravel()
    assert vector.size == 3

    skew_matrix = np.zeros((3, 3))
    skew_matrix[1, 0] = vector[2]
    skew_matrix[2, 0] = -vector[1]
    skew_matrix[0, 1] = -vector[2]
    skew_matrix[2, 1] = vector[0]
    skew_matrix[0, 2] = vector[1]
    skew_matrix[1, 2] = -vector[0]

    return skew_matrix


def skew_matrix2vector(matrix):
    assert matrix.shape == (3, 3)

    vector = np.zeros((3, 1))

    vector[0] = matrix[2, 1]
    vector[1] = -matrix[2, 0]
    vector[2] = matrix[1, 0]

    return vector


def get_rot_matrix_from_two_vectors(src, dst):
    __src = src / np.linalg.norm(src)
    __dst = dst / np.linalg.norm(dst)
    normal = np.cross(src, dst)
    if np.linalg.norm(normal) == 0:
        return np.eye(3)
    theta = np.arccos(np.clip(np.dot(__src, __dst), -1.0, 1.0))

    q = Quaternion(axis=normal, radians=theta)
    return q.rotation_matrix


def approxRotationMatrix(SE3_matrix):
    '''
    :param T: 4x4 3x4 matrix
    :return: T as SE3
    '''
    assert SE3_matrix.shape[1] == 4

    R = SE3_matrix[0:3, 0:3]
    t = SE3_matrix[0:3, 3]

    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    U, D, V = np.linalg.svd(R)
    scale = np.sum(D) / 3

    R = np.dot(U, np.eye(3))
    R = np.dot(R, V)

    t = t / scale

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def angle_between_vectors(vect_ref, vect):
    """
    This function returns the angle between two vectors
    :return:
    """

    c = np.dot(vect_ref.T, vect) / \
        (np.linalg.norm(vect_ref) * np.linalg.norm(vect))
    angle = np.arccos(np.clip(c, -1, 1))

    return angle


def evaluate_error_in_transformation(transform_est,
                                     transform_gt,
                                     degrees=True):
    """
    Return the angular error in rotation and translation as the error angle between two vector
    for both rotation and translation
    Ref:
    Fathian, et.al. (RAL 2018). QuEst: A Quaternion-Based Approach for Camera.
    Huynh, D. Q. (2009). Metrics for 3D Rotations: Comparison and Analysis.
    """
    assert transform_est.shape == transform_gt.shape == (4, 4)
    # ! Error in rotation

    error = 0.5 * (np.trace(transform_gt[0:3, 0:3].T.dot(
        transform_est[0:3, 0:3])) - 1)
    rot_err = np.arccos(np.clip(error, -1, 1)) / np.pi

    # ! Error in translation
    trans_err = angle_between_vectors(transform_gt[0:3, 3],
                                      transform_est[0:3, 3]) / np.pi

    dist_err = np.linalg.norm(transform_gt[0:3, 3] - transform_est[0:3, 3])

    if degrees:
        return np.degrees(rot_err), np.degrees(trans_err), dist_err
    else:
        return np.abs(rot_err), np.abs(trans_err), dist_err


def uniform_mask_sampling(h, w, stride=5):
    u = np.linspace(0, w - 1, w//stride).astype(int)
    v = np.linspace(0, h - 1, h//stride).astype(int)
    uu, vv = np.meshgrid(u, v)
    mask = np.zeros((h, w), dtype=bool)
    mask[vv.flatten(), uu.flatten()] = True
    return mask


def triangulate_points_from_cam_pose(cam_pose, x1, x2):
    '''
    Triangulate 4D-points based on the relative camera pose and pts1 & pts2 matches
    :param Mn: Relative pose (4, 4) from cam1 to cam2
    :param x1: (3, n)
    :param x2: (3, n)
    :return:
    '''

    assert x1.shape[0] == 3
    assert x1.shape == x2.shape

    cam_pose = np.linalg.inv(cam_pose)
    landmarks_x1 = []
    for p1, p2 in zip(x1.T, x2.T):
        p1x = vector2skew_matrix(p1.ravel())
        p2x = vector2skew_matrix(p2.ravel())

        A = np.vstack(
            (np.dot(p1x,
                    np.eye(4)[0:3, :]), np.dot(p2x, cam_pose[0:3, :])))
        U, D, V = np.linalg.svd(A)
        landmarks_x1.append(V[-1])

    landmarks_x1 = np.asarray(landmarks_x1).T
    landmarks_x1 = landmarks_x1 / landmarks_x1[3, :]
    return landmarks_x1[:3, :]  # Return only the 3D points
