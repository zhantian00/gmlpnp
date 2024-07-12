from typing import Tuple, List
import time, os
import numpy as np 
import cv2
from scipy.spatial.transform import Rotation
import yaml
from generate_synthetic_data import generate_synthetic_data
import matplotlib.pyplot as plt
import seaborn as sns

import build.advancedpnp as advpnp

# load config
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

CAMERA_INTRINSICS = np.array(cfg['CAMERA_INTRINSICS'])
NUM_REPEAT = 500
line_styles = {
    'EPnP': '-s',
    'BA': '-o',
    'PPnP': '-X',
    'DLT': '-h',
    'SQPnP': '-D',
    'BA+RANSAC': '-p',
    'UPnP': '-P',
    'CPnP': '-<',
    'EPnPU': '-|',
    'DLSU': '-H',
    'REPPnP': '-1',
    'MLPnP': '-x',
    'GMLPnP*': '--2',
    'GMLPnP': '-^',
}
UNIFIED_METHOD_NAMES = ['MLPnP', 'UPnP', 'GMLPnP*', 'GMLPnP', 'GMLPnP_WP']
METHOD_NAMES = ['EPnP', 'BA', 'PPnP', 'SQPnP', 'UPnP', 'CPnP', 'MLPnP', 'GMLPnP*', 'GMLPnP']

sns.set_palette("tab10", n_colors=len(line_styles))
color_palette = sns.color_palette()
color_palette = {l: c for l, c in zip(list(line_styles), color_palette)}
color_palette['GMLPnP*'] = 'red'
color_palette['GMLPnP'] = 'red'


def eval_r(r_vec_gt: np.ndarray, r_vec_est: np.ndarray):
    """Evaluate Rotation error. 
    Input a batch of data or a single item
    :param r_vec_gt: ground truth of rotation vector
    :param r_vec_est: estimation of rotation vector
    :return r_error: the RMS error of Rotation in degrees
    """
    r_vec_gt = np.array(r_vec_gt); r_vec_est = np.array(r_vec_est)
    r_gt = Rotation.from_rotvec(r_vec_gt).as_matrix()
    r_est = Rotation.from_rotvec(r_vec_est).as_matrix()
    cos_error = np.einsum('ijk,ijk->ik', r_gt, r_est)
    cos_error = np.clip(cos_error, -1.0, 1.0)
    anglar_error = np.rad2deg(np.arccos(cos_error))
    msre_per_row = np.max(anglar_error, axis=1)
    r_error = np.mean(msre_per_row)
    return r_error


def eval_t(t_vec_gt: np.ndarray, t_vec_est: np.ndarray):
    """Evaluate translation error, in mean percentage squared root error.
    Input a batch of data or a single item.
    :return t_error: the MPSRE of translation in percentage
    """
    t_vec_gt = np.array(t_vec_gt); t_vec_est = np.array(t_vec_est)
    # the mean squared root error per row
    msre_per_row = np.linalg.norm(t_vec_gt - t_vec_est, ord=2, axis=1)
    # the msrs / translation length
    mpsre_per_row = msre_per_row / np.linalg.norm(t_vec_gt, ord=2, axis=1)
    t_error = np.mean(mpsre_per_row) * 100
    return t_error

def image2ray(image_points, camera_mat):
    """For perspective camera, lift the the image points to unit sphere
    to get the poejection ray.
    Input points should shaped (n, 2),
    Output ray shaped (n, 3)
    """
    image_points_homo = np.column_stack([image_points, np.ones(len(image_points))])
    points_inv_proj = np.linalg.inv(camera_mat) @ image_points_homo.T
    norms = np.linalg.norm(points_inv_proj, axis=0)
    proj_ray = points_inv_proj / norms
    return proj_ray.T

    
def solve_pnp(method_name,
              points_3d, 
              points_proj,
              camera_mat=CAMERA_INTRINSICS,
              rvec_init=np.zeros(3),
              tvec_init=np.zeros(3),
              info_mat=np.identity(3),
              cov_obj=np.identity(3),
              cov_img=np.identity(2)):
    """For algorithms that only works for perpective camera,
    pass the camera (intrinsics) matrix which is a 3x3 matrix, 
    and the projected image points should be rectified.
    For unified pnp algorthms, pass the projection ray to the param `points_proj`,
    and the `camera_mat` param is not needed.
    Which means if the user uses the data from a perspective camera while he 
    wants to use unified algorithm, he must lift the image points to unit sphere
    before calling this function. 
    """
    if method_name == 'BA':
        success, r_est, t_est = cv2.solvePnP(
            points_3d, 
            points_proj, 
            cameraMatrix=camera_mat,
            distCoeffs=np.zeros(4),
            # rvec=rvec_init,
            # tvec=tvec_init,
            # useExtrinsicGuess=use_init_guess,
            flags=cv2.SOLVEPNP_ITERATIVE) 
    elif method_name == 'BA_g2o':
        success, r_est, t_est = advpnp.solve_pnp_ba(
            points_3d, 
            points_proj, 
            camera_mat,
            rvec_init, tvec_init,
            )
    elif method_name == 'EPnP':
        success, r_est, t_est = cv2.solvePnP(
            points_3d, 
            points_proj, 
            cameraMatrix=camera_mat,
            distCoeffs=np.zeros(4), 
            flags=cv2.SOLVEPNP_EPNP) 
    elif method_name == 'SQPnP':
        success, r_est, t_est = cv2.solvePnP(
            points_3d, 
            points_proj, 
            cameraMatrix=camera_mat,
            distCoeffs=np.zeros(4), 
            flags=cv2.SOLVEPNP_SQPNP) 
    elif method_name == 'PPnP':
        success, r_est, t_est = advpnp.solve_pnp_ppnp(
            points_3d, 
            points_proj,
            camera_mat,
            rvec_init, tvec_init,
            1e-5)
        
    elif method_name == 'DLT':
        r_est, t_est = advpnp.solve_pnp_dlt(
            points_3d, 
            points_proj)

    elif method_name == 'CPnP':
        r_est_gn, t_est_gn = advpnp.solve_pnp_cpnp(points_proj,
                              points_3d,
                              np.array([camera_mat[0,0], camera_mat[1,1], camera_mat[0,2], camera_mat[1,2]])
                              )
        r_est = Rotation.from_quat(r_est_gn).as_rotvec()
        t_est = t_est_gn

    elif method_name == 'MLPnP':
        r_est, t_est = advpnp.solve_pnp_mlpnp(
            points_3d,
            points_proj,
            info_mat,
        )

    elif method_name == 'UPnP':
        r_est, t_est = advpnp.solve_pnp_upnp(
            points_3d,
            points_proj,
        )

    elif method_name == 'GMLPnP*':
        success, r_est, t_est = advpnp.solve_pnp_gmlpnp_with_prior(
            points_3d, 
            points_proj,
            rvec_init, tvec_init,  
            info_mat,
            False,
            )
    elif method_name == 'GMLPnP':
        success, r_est, t_est = advpnp.solve_pnp_gmlpnp(
            points_3d, 
            points_proj,
            rvec_init, tvec_init,  
            np.identity(3),
            False,
            )
    return r_est, t_est


def eval_from_synthetic_data(method_name: str, 
              num_points: int, 
              noise_std: float) -> Tuple:
    """Solve pnp with different algorithms using synthetic data, 
    repeat the process by NUM_REPEAT times.
    :param method_name: algorithm name
    :param num_points: the number of points generated
    :param point_noise_std: the noise standard deviation 
    :return r_msre, t_mpsre: the Rotation and translation error
    """
    np.random.seed(56)
    r_est, r_gt = np.zeros((NUM_REPEAT, 3)), np.zeros((NUM_REPEAT, 3))
    t_est, t_gt = np.zeros((NUM_REPEAT, 3)), np.zeros((NUM_REPEAT, 3))
    exec_time = 0
    for i in range(NUM_REPEAT):
        points_3d, points_proj, r_gt[i], t_gt[i], cov_full, cov_obj, cov_img = generate_synthetic_data(num_points, noise_std, 10*noise_std)
        info_mat = np.linalg.inv(cov_full)
        rvec_init = r_gt[i] #+ np.random.normal(0, 0.1, 3)
        tvec_init = t_gt[i] #+ np.random.normal(0, 0.1, 3)
        if method_name in UNIFIED_METHOD_NAMES:
            points_proj = image2ray(points_proj, CAMERA_INTRINSICS)
        tic = time.time()
        r_est_, t_est_ = solve_pnp(method_name, 
                                    points_3d, 
                                    points_proj, 
                                    CAMERA_INTRINSICS,
                                    info_mat=info_mat,
                                    cov_obj=cov_obj, 
                                    cov_img=cov_img)
        r_est[i], t_est[i] = r_est_.flatten(), t_est_.flatten()
        if np.linalg.norm(t_est[i]-t_gt[i]) > np.linalg.norm(t_gt[i]):
            r_est_, t_est_ = np.zeros(3), np.zeros(3)
        toc = time.time()
        exec_time = exec_time + toc - tic
    # print(f"r_gt {r_gt}, t_gt {t_gt}")
    # print(f"r_est {r_est}, t_est {t_est}")
    r_msre = eval_r(r_gt, r_est)
    t_mpsre = eval_t(t_gt, t_est)
    return r_msre, t_mpsre, exec_time/NUM_REPEAT*1000


def iter_over_num(method_names: str, 
                  noise_std: float):
    """Iterate over method by number of points and draw figure.
    :param point_noise_std: the 3d points noise (standard deviation) in meter
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, method_name in enumerate(method_names):
        x_axis_list, r_msre_list, t_mpsre_list, time_list = [], [], [], []
        for num_points in range(20, 210, 10):
            r_msre, t_mpsre, exec_time = eval_from_synthetic_data(method_name, 
                                                                  num_points, 
                                                                  noise_std)
            print(f"num={num_points} {method_name}: Rotation -> Error {r_msre} degree, Translation -> Error {t_mpsre} %")
            x_axis_list.append(num_points)
            r_msre_list.append(r_msre)
            t_mpsre_list.append(t_mpsre)
            time_list.append(exec_time)
        # Subplot 1
        axes[0].plot(x_axis_list, r_msre_list, line_styles[method_name], label=method_name, color=color_palette[method_name])
        # Subplot 2
        axes[1].plot(x_axis_list, t_mpsre_list, line_styles[method_name], label=method_name, color=color_palette[method_name])

    # Set subplot 1
    axes[0].set_ylim(0.2, 1.4)
    axes[0].set_xlabel('Number of Points')
    axes[0].set_ylabel('Rotation Error (degrees)')
    axes[0].set_title('Rotation Error vs #Points')
    # axes[0].legend()
    axes[0].grid(True)

    # Set grid density for the first subplot
    axes[0].locator_params(nbins=10)  # Adjust the value (e.g., 5) as needed

    # Set subplot 2
    axes[1].set_ylim(0.1, 1.4)
    axes[1].set_xlabel('Number of Points')
    axes[1].set_ylabel('Translation Error (%)')
    axes[1].set_title('Translation Error vs #Points')
    # axes[1].legend()
    axes[1].grid(True)

    # Set grid density for the second subplot
    axes[1].locator_params(nbins=10)  # Adjust the value (e.g., 5) as needed

    lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor =(0.5,0.0))
    plt.subplots_adjust(top=0.88, bottom=0.22, right=0.9, left=0.125, hspace=0.2, wspace=0.2)
    plt.savefig('error_vs_num_of_points.pdf')
    # plt.show()
    plt.close()
    return


def iter_over_noise(method_names: List, num_points: int = 30):
    """Iterate over methods by noise covariance determinant and draw figure.
    :method_names: the list of algorithm names
    :param num_points: the number of generate point pairs
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, method_name in enumerate(method_names):
        x_axis_list, r_msre_list, t_mpsre_list = [], [], []
        for noise_std in np.arange(0.02, 0.52, 0.02, dtype=np.float32):
            r_msre, t_mpsre, exec_time = eval_from_synthetic_data(method_name, num_points, noise_std)
            print(f"noise={noise_std} {method_name}: Rotation -> MSRE {r_msre} degree, Translation -> MPSRE {t_mpsre} %")
            x_axis_list.append(noise_std)
            r_msre_list.append(r_msre)
            t_mpsre_list.append(t_mpsre)

        # Subplot 1
        axes[0].plot(x_axis_list, r_msre_list, line_styles[method_name], label=method_name, color=color_palette[method_name])
        # Subplot 2
        axes[1].plot(x_axis_list, t_mpsre_list, line_styles[method_name], label=method_name, color=color_palette[method_name])

    # Set subplot 1
    axes[0].set_ylim(0, 7)
    axes[0].set_xlabel('Noise Standard Deviation (m)')
    axes[0].set_ylabel('Rotation Error (degree)')
    axes[0].set_title('Rotation Error vs Noise')
    # axes[0].legend()
    axes[0].grid(True)
    # Set grid density for the first subplot
    axes[0].locator_params(nbins=10)  # Adjust the value (e.g., 5) as needed

    # Set subplot 2
    axes[1].set_ylim(0, 5)
    axes[1].set_xlabel('Noise Standard Deviation (m)')
    axes[1].set_ylabel('Translation Error (%)')
    axes[1].set_title('Translation Error vs Noise')
    # axes[1].legend()
    axes[1].grid(True)

    # Set grid density for the second subplot
    axes[1].locator_params(nbins=10)  # Adjust the value (e.g., 5) as needed

    lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor =(0.5,0.0))
    plt.subplots_adjust(top=0.88, bottom=0.22, right=0.9, left=0.125, hspace=0.2, wspace=0.2)
    plt.savefig('error_vs_noise.pdf')
    # plt.show()
    plt.close()
    return


def main():
    method_names = METHOD_NAMES
    iter_over_noise(method_names, num_points=50)
    iter_over_num(method_names, noise_std=0.1)
    return


if __name__ == "__main__":
    main()