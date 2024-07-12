#include "gmlpnp_with_prior.h"
#include "mlpnp.h"


std::tuple<Eigen::MatrixXd, Eigen::Vector3d, Eigen::Vector3d> solvePnpByGmlpnpWP(
    const Eigen::Ref<const MatrixX3d>& object_points,
    const Eigen::Ref<const MatrixX3d>& proj_ray,
    const Eigen::Ref<const Eigen::Vector3d>& r_vec,
    const Eigen::Ref<const Eigen::Vector3d>& t_vec,
    const Eigen::Ref<const Eigen::Matrix3d>& info_mat,
    const bool use_init_guess) {

    Eigen::Vector3d r_vec_init, t_vec_init;
    if (use_init_guess) {
        r_vec_init = r_vec;
        t_vec_init = t_vec;
    }
    else {
        // initial guess with MLPnP
        mlpnp(object_points, proj_ray, info_mat.inverse(), false, r_vec_init, t_vec_init);
    }

    g2o::SparseOptimizer optimizer;
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    
    // define pose vertex
    g2o::VertexSE3Expmap* pose_vertex = new g2o::VertexSE3Expmap();
    //convert rotation vecter to matrix
    Eigen::AngleAxisd r_vec_init_w2c(r_vec_init.norm(), 
                                     r_vec_init.normalized());
    Eigen::Matrix3d r_mat_init_c2w = r_vec_init_w2c.matrix().transpose();
    Eigen::Vector3d t_vec_init_c2w = -r_mat_init_c2w * t_vec_init;
    pose_vertex -> setId(0);
    pose_vertex -> setEstimate(g2o::SE3Quat(r_mat_init_c2w, t_vec_init_c2w));
    optimizer.addVertex(pose_vertex);

    // init scale
    const int num_points = object_points.rows();

    // define 3d point vertex and projection edge
    int vertex_id = 1; // id=0 for pose_vertex, so we allocate from 1
    for (int row = 0; row < num_points; ++row) {
        g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
        point_vertex -> setId(vertex_id);
        point_vertex -> setMarginalized(true);
        point_vertex -> setFixed(true);
        point_vertex -> setEstimate(object_points.row(row));
        if (!optimizer.addVertex(point_vertex)) {
            assert(false);
        }
        vertex_id++;

        EdgeReconstructionUV2XYZ* project_edge = 
            new EdgeReconstructionUV2XYZ();
        project_edge -> setVertex(
            0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose_vertex)
        );
        project_edge -> setVertex(
            1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(point_vertex)
        );
        project_edge -> setMeasurement(proj_ray.row(row));
        project_edge -> information() = info_mat;
        project_edge -> setParameterId(0, 0);
        if (!optimizer.addEdge(project_edge)) {
            assert(false);
        }
    }
    
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(3);

    Eigen::MatrixXd residual = Eigen::MatrixXd(num_points, 3);
    int data_index = 0;
    for (auto& edge : optimizer.edges()) {
        EdgeReconstructionUV2XYZ* edge_ptr = dynamic_cast<EdgeReconstructionUV2XYZ*>(edge);
        residual.row(data_index) = edge_ptr -> error();
        data_index++;
    }

    Eigen::AngleAxisd r_vec_in_angleaxis(pose_vertex->estimate().rotation());
    Eigen::Vector3d r_vec_out, t_vec_out;
    r_vec_out = -r_vec_in_angleaxis.angle() * r_vec_in_angleaxis.axis();
    t_vec_out = -pose_vertex->estimate().rotation().matrix().transpose() 
        * pose_vertex->estimate().translation();
    return std::make_tuple(residual, r_vec_out, t_vec_out);
}