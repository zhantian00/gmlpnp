#pragma once

#include <iostream>
#include <vector>
#include <tuple>

#include "opengv/absolute_pose/CentralAbsoluteAdapter.hpp"
#include "opengv/absolute_pose/methods.hpp"
#include "opengv/types.hpp"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"

using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;


std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnpByUpnp(
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const MatrixX3d>&);


std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnpByMlpnp(
    const Eigen::Ref<const MatrixX3d>& object_points,
    const Eigen::Ref<const MatrixX3d>& proj_ray,
    const Eigen::Ref<const Eigen::Matrix3d>& info_mat);