from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


# load config
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

IMAGE_SIZE = cfg['IMAGE_SIZE']
CAMERA_INTRINSICS = cfg['CAMERA_INTRINSICS']
POINT_RANGE = cfg['POINT_RANGE']


def oulier_limitation(array, thres):
    norm = np.linalg.norm(array, axis=1)
    res = np.array([a * thres / n if n > thres else a for n, a in zip(norm, array)])
    return res


# TODO: add visiable constraint, reject points outside camera resolution
def generate_synthetic_data(num_points: int, object_noise_std: float, image_noise_std: float) -> Tuple:
    """Given the number of points and randomly generate a set of pairs of 3D
    and 2D projection points. The 3D points are pertubed with gaussian noise. 
    :param num_points: number of points
    :param noise_std: the standard deviation of 3D point noise
    :return points_3d_With_noise: 3D points in world coordinate with noise, shaped (num_points, 3)
    :return points_proj: corresponding 2D projection, shaped (num_points, 2)
    :return r_vec: camera orientation in world coordinate in rotation vector, shaped (3)
    :return t_vec: camera translation vector in world coordinate, shaped (3)
    """
    # generate random camera pose
    random_rotation = Rotation.random()
    r_mat = random_rotation.as_matrix()
    r_vec = random_rotation.as_rotvec()
    t_vec = np.mean(POINT_RANGE, axis=0)
    # t_vec = np.random.uniform([-1, -1, -1], 
    #                           [1, 1, 1], 3) * 10

    # generate random points in camera coordinate, 
    # we do so to make sure the depth is positive
    points_3d_cam = np.random.uniform(POINT_RANGE[0], POINT_RANGE[1],
                                      (num_points, 3))
    points_proj = (np.array(CAMERA_INTRINSICS) @ points_3d_cam.T).T
    points_proj = points_proj[:, :2] / points_proj[:, 2:]
    # get these point in world coordinate by convert it back and add noise
    points_3d_world = r_mat.T @ (points_3d_cam.T - t_vec[:, None])
    points_3d_world = points_3d_world.T

    object_noise = np.array([1.0, np.random.rand(), np.random.rand()])
    cov_obj = np.diag(np.power(object_noise_std * object_noise, 2))
    random_cov_rot = Rotation.random()
    r_cov_mat = random_cov_rot.as_matrix()
    cov_obj = r_cov_mat @ cov_obj @ r_cov_mat.T
    object_noise = np.random.multivariate_normal([0, 0, 0], cov_obj, num_points)
    object_noise = oulier_limitation(object_noise, 3 * object_noise_std)

    image_noise = np.array([1.0, np.random.rand()])
    cov_img = np.diag(np.power(image_noise_std * image_noise, 2))
    image_noise = np.random.multivariate_normal([0, 0], cov_img, num_points)
    image_noise = oulier_limitation(image_noise, 3 * image_noise_std)

    points_3d_with_noise = points_3d_world + object_noise
    points_proj = points_proj + image_noise
    inverse_proj = np.linalg.inv(np.array(CAMERA_INTRINSICS))

    cov_img_3 = np.zeros((3,3))
    cov_img_3[:2, :2] = cov_img
    cov_full = r_mat @ inverse_proj @ cov_img_3 @ inverse_proj.T @ r_mat.T + cov_obj

    return points_3d_with_noise, points_proj, r_vec, t_vec, cov_full, cov_obj, cov_img


def generate_synthetic_data_from_det(num_points: int, covariance_det: float) -> Tuple:
    # generate random camera pose
    random_rotation = Rotation.random()
    r_mat = random_rotation.as_matrix()
    r_vec = random_rotation.as_rotvec()
    t_vec = np.mean(POINT_RANGE, axis=0)
    # generate random points in camera coordinate, 
    # we do so to make sure the depth is positive
    points_3d_cam = np.random.uniform(POINT_RANGE[0], POINT_RANGE[1],
                                      (num_points, 3))
    points_proj = (np.array(CAMERA_INTRINSICS) @ points_3d_cam.T).T
    points_proj = points_proj[:, :2] / points_proj[:, 2:]
    # get these point in world coordinate by convert it back and add noise
    points_3d_world = r_mat.T @ (points_3d_cam.T - t_vec[:, None])
    points_3d_world = points_3d_world.T

    mean = [0, 0, 0]
    while(True):
        covariance_matrix = np.random.randn(3, 3)
        # Make it positive definite and symmetric
        covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T)
        # Normalize to ensure determinant is 1
        det = np.linalg.det(covariance_matrix)
        covariance_matrix /= np.power(det, 1.0/3.0)
        eig_values, eig_vecs = np.linalg.eig(covariance_matrix)
        if np.max(eig_values) / np.min(eig_values) < 1000.: break

    # Scale up to the determinant
    covariance_matrix *= np.power(covariance_det, 1.0/3.0)
    noise = np.random.multivariate_normal(mean, 
                                          covariance_matrix, 
                                          num_points)
    points_3d_with_noise = points_3d_world + noise
    return points_3d_with_noise, points_proj, r_vec, t_vec, covariance_matrix
