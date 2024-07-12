#include "opengv_wrapper.h"

std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnpByUpnp(
    const Eigen::Ref<const MatrixX3d>& object_points,
    const Eigen::Ref<const MatrixX3d>& proj_ray) {
    opengv::bearingVectors_t bearing_vectors;
    opengv::points_t points;
    size_t num_points = object_points.rows();
    for (size_t i = 0; i < num_points; ++i) {
        bearing_vectors.push_back(proj_ray.row(i));
        points.push_back(object_points.row(i));
    }
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(
        bearing_vectors, points);

    opengv::transformations_t transformations;
    transformations = opengv::absolute_pose::upnp(adapter);

    // TODO: WTF is this, figure out why UPnP return a vector.
    opengv::transformation_t transformation = transformations[0];
    Eigen::Matrix3d rotation_matrix = transformation.block<3, 3>(0, 0);
    Eigen::AngleAxisd angleaxis(rotation_matrix);
    Eigen::Vector3d r_vec_out = - angleaxis.angle() * angleaxis.axis();
    Eigen::Vector3d t_vec_out = - rotation_matrix.transpose() * transformation.block<3, 1>(0, 3);
    return std::make_tuple(r_vec_out, t_vec_out);
}


std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnpByMlpnp(
    const Eigen::Ref<const MatrixX3d>& object_points,
    const Eigen::Ref<const MatrixX3d>& proj_ray,
    const Eigen::Ref<const Eigen::Matrix3d>& info_mat) {
    opengv::bearingVectors_t bearing_vectors;
    opengv::points_t points;
    opengv::cov3_mats_t conv_mats;
    size_t num_points = object_points.rows();
    opengv::cov3_mat_t cov_mat = info_mat.inverse();
    for (size_t i = 0; i < num_points; ++i) {
        bearing_vectors.push_back(proj_ray.row(i));
        points.push_back(object_points.row(i));
        conv_mats.push_back(cov_mat);
    }
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(
        bearing_vectors, points, conv_mats);

    opengv::transformation_t transformation = opengv::absolute_pose::mlpnp(adapter);

    Eigen::Matrix3d rotation_matrix = transformation.block<3, 3>(0, 0);
    Eigen::AngleAxisd angleaxis(rotation_matrix);
    Eigen::Vector3d r_vec_out = - angleaxis.angle() * angleaxis.axis();
    Eigen::Vector3d t_vec_out = - rotation_matrix.transpose() * transformation.block<3, 1>(0, 3);
    return std::make_tuple(r_vec_out, t_vec_out);
}