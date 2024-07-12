#include "ppnp.h"

std::tuple<bool, Eigen::Vector3d, Eigen::Vector3d> solvePnpWithPpnp(
    const Eigen::Ref<const MatrixX3d>& object_points,
    const Eigen::Ref<const MatrixX2d>& image_points,
    const Eigen::Ref<const Eigen::Matrix3d>& camera_matrix,
    const Eigen::Ref<const Eigen::Vector3d>& r_vec_init=Eigen::Vector3d::Zero(),
    const Eigen::Ref<const Eigen::Vector3d>& t_vec_init=Eigen::Vector3d::Zero(),
    double tol=1e-3) {
    // the procedure here follows the original paper
    const int n = object_points.rows();
    Eigen::MatrixXd P(n, 3);
    P.leftCols(2) = image_points;
    P.col(2).setOnes();
    P = (camera_matrix.inverse() * P.transpose()).transpose(); // TODO: optimize inverse
    MatrixX3d S = object_points;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) - (e * e.transpose()) / n;
    Eigen::VectorXd II = e / n;
    double err = 999;
    Eigen::MatrixXd E_old = 1000.0 * Eigen::MatrixXd::Ones(n, 3);

    Eigen::AngleAxisd r_vec(r_vec_init.norm(), r_vec_init.normalized());
    Eigen::Matrix3d R = r_vec.matrix();
    Eigen::Vector3d c;
    Eigen::Matrix3d U, VT;
    Eigen::MatrixXd PR(n, 3), Y(n, 3), E(n, 3);
    Eigen::VectorXd P_squred_sum(n), Zmindiag(n);

    // init depth
    c = -R.transpose() * t_vec_init;
    Y = S - e * c.transpose();
    Zmindiag = (P * R * Y.transpose()).diagonal();
    P_squred_sum = P.array().square().rowwise().sum();
    Zmindiag = Zmindiag.cwiseQuotient(P_squred_sum);
    Zmindiag = Zmindiag.cwiseMax(0.0);
    Z = Zmindiag.asDiagonal();

    int iter = 0;
    int max_iter = 30;
    bool success_flag = true;
    while (err > tol && iter < max_iter) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            P.transpose() * Z * A * S, 
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        VT = svd.matrixV().transpose();
        R = U * Eigen::Vector3d(
            1.0, 1.0, (U * VT).determinant()).asDiagonal() * VT;
        PR = P * R;
        c = (S - Z * PR).transpose() * II;
        Y = S - e * c.transpose();
        Zmindiag = (PR * Y.transpose()).diagonal();
        Zmindiag = Zmindiag.cwiseQuotient(P_squred_sum);
        Zmindiag = Zmindiag.cwiseMax(0.0);
        Z = Zmindiag.asDiagonal();
        E = Y - Z * PR;
        err = (E - E_old).norm();
        E_old = E;
        iter++;
    }

    Eigen::Vector3d t_vec_out = -R * c;
    // get the rotation vector from rotation matrix
    Eigen::AngleAxisd rotation_vector(R);
    Eigen::Vector3d r_vec_out = rotation_vector.axis() * rotation_vector.angle();
    return std::make_tuple(success_flag, r_vec_out, t_vec_out);
}
