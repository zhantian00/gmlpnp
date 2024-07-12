#pragma once
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <tuple>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/solver.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/io_helper.h"
#include "g2o/core/parameter.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_fixed_sized_edge.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"

using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

class EdgeReconstructionUV2XYZ : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
    g2o::VertexSE3Expmap, g2o::VertexPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeReconstructionUV2XYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError();
    void updateScale();
public:
    double _scale;
};

std::tuple<std::vector<Eigen::MatrixXd>, Eigen::Vector3d, Eigen::Vector3d> solvePnpByGmlpnp(
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const MatrixX3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Vector3d>&,
    const Eigen::Ref<const Eigen::Matrix3d>&,
    const bool);