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


// DLT method to solve PnP problem. Note that if all points are on a plane (e.g. calibration board), the solution is not unique.
// @article{abdel2015direct,
//   title={Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry},
//   author={Abdel-Aziz, Yousset I and Karara, Hauck Michael and Hauck, Michael},
//   journal={Photogrammetric engineering \& remote sensing},
//   volume={81},
//   number={2},
//   pages={103--107},
//   year={2015},
//   publisher={Elsevier}
// }
// This impl is borrowed from https://github.com/qiaozhijian/PnP-Solver/blob/main/include/pnp_solver.h
std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnPbyDLT(
    const Eigen::Ref<const MatrixX3d>& object_points, 
    const Eigen::Ref<const MatrixX2d>& image_points);