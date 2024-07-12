#pragma once
#include "gmlpnp.h"


std::tuple<Eigen::MatrixXd, Eigen::Vector3d, Eigen::Vector3d> solvePnpByGmlpnpWP(
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Matrix3d>&,
    const bool use_init_guess);