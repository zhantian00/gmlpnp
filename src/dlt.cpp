#include "dlt.h"

std::tuple<Eigen::Vector3d, Eigen::Vector3d> solvePnPbyDLT(
    const Eigen::Ref<const MatrixX3d>& object_points, 
    const Eigen::Ref<const MatrixX2d>& image_points) {

    const int n = object_points.rows();
    Eigen::MatrixXd A(n * 2, 12);
    for (int i = 0; i < n; i++) {
        A.row(i * 2) << object_points(i, 0), object_points(i, 1), object_points(i, 2), 1, 0, 0, 0, 0, -image_points(i, 0) * object_points(i, 0), -image_points(i, 0) * object_points(i, 1), -image_points(i, 0) * object_points(i, 2), -image_points(i, 0);
        A.row(i * 2 + 1) << 0, 0, 0, 0, object_points(i, 0), object_points(i, 1), object_points(i, 2), 1, -image_points(i, 1) * object_points(i, 0), -image_points(i, 1) * object_points(i, 1), -image_points(i, 1) * object_points(i, 2), -image_points(i, 1);
    }
    // compute the rank of A
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(11);
    Eigen::Matrix<double, 3, 4> H_hat;
    H_hat << h(0), h(1), h(2), h(3),
            h(4), h(5), h(6), h(7),
            h(8), h(9), h(10), h(11);
    double s = H_hat(2, 0) * object_points(0, 0) + H_hat(2, 1) * object_points(0, 1) + H_hat(2, 2) * object_points(0, 3) + H_hat(2, 3);
    if (s < 0) { // make sure the depth is positive
        H_hat = -H_hat;
    }
    // if this value is too big, the solution is not reliable (since the residual equals to the smallest singular value)
    double smallest_singular_value = svd.singularValues()(svd.singularValues().size() - 1);
    //std::cout << "smallest_singular_value: " << smallest_singular_value << std::endl;
    Eigen::Matrix3d R_hat = H_hat.block<3, 3>(0, 0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(R_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd_R.matrixU();
    Eigen::Matrix3d V = svd_R.matrixV();

    Eigen::Matrix3d R_cw;
    Eigen::Vector3d t_cw;
    R_cw = U * V.transpose();
    if (R_cw.determinant() < 0) { // make sure the rotation matrix is right-handed
        V.col(2) = -V.col(2);
        R_cw = U * V.transpose();
    }
    t_cw = H_hat.col(3) / H_hat.col(0).norm();
    Eigen::AngleAxisd rotation_vector(R_cw);
    Eigen::Vector3d r_vec_out = rotation_vector.axis() * rotation_vector.angle();
    return std::make_tuple(r_vec_out, t_cw);
}