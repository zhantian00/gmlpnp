#pragma once

#include <iostream>
#include <tuple>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"

using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

// Implementation of PPnP
// @inproceedings{garro2012solving,
//     title={Solving the pnp problem with anisotropic orthogonal procrustes analysis},
//     author={Garro, Valeria and Crosilla, Fabio and Fusiello, Andrea},
//     booktitle={2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization \& Transmission},
//     pages={262--269},
//     year={2012},
//     organization={IEEE}
//     }
// :param object_points: the 3D points in world coordinate
// :param image_points: the corresponding 2D projection in camera
// :param camera_matrix: camera intrinsics
// :param r_vec_init: init rotation vector, default to be zero vector
// :param t_vec_init: init translation vector, default to be zero vector
// :param tol: exit threshold
// :return success_flag: if the program succeeds
// :return r_vec: rotation vector
// :return t_vec: translation vector
std::tuple<bool, Eigen::Vector3d, Eigen::Vector3d> solvePnpWithPpnp(
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const MatrixX2d>&,
    const Eigen::Ref<const Eigen::Matrix3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    double);
