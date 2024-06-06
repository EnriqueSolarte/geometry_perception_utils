import numpy as np
from geometry_perception_utils.geometry_utils import vector2skew_matrix, approxRotationMatrix

'''
Reference:
1. https://www-users.cs.york.ac.uk/~wsmith/papers/TIP_SFM.pdf
2. https://github.com/acvictor/DLT/blob/master/DLT.py
3. https://www.youtube.com/watch?v=RR8WXL-kMzA
'''


class PnP:
    """
    PnP solver suing Linear Least Squares (Linear approximation)
    """

    def __init__(self):
        pass

    def build_matrix_A(self, bearings, xyz):
        def compute_matrix_A_i(x, w):
            x_skew = vector2skew_matrix(x)
            w_expand = np.array([
                # ! w_i matrix definition
                [w[0], w[1], w[2], 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, w[0], w[1], w[2], 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, w[0], w[1], w[2], 1]
            ])
            return x_skew @ w_expand

        list_A = []
        for x, w in zip(bearings.T, xyz.T):
            A = compute_matrix_A_i(w=w, x=x)
            list_A.append(A)
        return np.vstack(list_A)

    def recover_pose(self, bearings, landmarks):
        assert bearings.shape[1] == landmarks.shape[1]
        assert bearings.shape[0] == 3
        assert landmarks.shape[0] == 3

        # * Ab = 0
        A = self.build_matrix_A(bearings, landmarks)
        _, _, V = np.linalg.svd(A)

        # Linear solution
        b = V[-1]

        b = b.reshape(3, 4)

        # * b = [ R | t ] --> Force R into SO(3)
        return approxRotationMatrix(b)
