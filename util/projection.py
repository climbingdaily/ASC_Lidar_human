import numpy as np


def world_to_camera(X, extrinsic_matrix):
    n = X.shape[0]
    X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    X = np.dot(extrinsic_matrix, X).T
    return X[..., :3]


def camera_to_pixel(X, intrinsic_matrix, distortion_coefficients):
    # focal length
    f = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    # center principal point
    c = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c
