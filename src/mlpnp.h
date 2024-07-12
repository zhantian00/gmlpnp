#pragma once

#include <stdlib.h>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** A 3-vector of unit length used to describe landmark observations/bearings
 *  in camera frames (always expressed in camera frames)
 */
typedef Eigen::Vector3d
    bearingVector_t;

/** An array of bearing-vectors */
typedef std::vector<bearingVector_t, Eigen::aligned_allocator<bearingVector_t> >
    bearingVectors_t;

/** A 3-vector describing a translation/camera position */
typedef Eigen::Vector3d
    translation_t;

/** An array of translations */
typedef std::vector<translation_t, Eigen::aligned_allocator<translation_t> >
    translations_t;

/** A rotation matrix */
typedef Eigen::Matrix3d
    rotation_t;

/** An array of rotation matrices as returned by fivept_kneip [7] */
typedef std::vector<rotation_t, Eigen::aligned_allocator<rotation_t> >
    rotations_t;

/** A 3x4 transformation matrix containing rotation \f$ \mathbf{R} \f$ and
 *  translation \f$ \mathbf{t} \f$ as follows:
 *  \f$ \left( \begin{array}{cc} \mathbf{R} & \mathbf{t} \end{array} \right) \f$
 */
typedef Eigen::Matrix<double,3,4>
    transformation_t;

/** An array of transformations */
typedef std::vector<transformation_t, Eigen::aligned_allocator<transformation_t> >
    transformations_t;

/** A 3-matrix containing the 3D covariance information of a bearing vector */
typedef Eigen::Matrix3d
cov3_mat_t;

/** A 2-matrix containing the 2D covariance information of a bearing vector
*/
typedef Eigen::Matrix2d
cov2_mat_t;

/** An array of 2D covariance matrices */
typedef std::vector<cov2_mat_t, Eigen::aligned_allocator<cov2_mat_t> >
cov2_mats_t;

/** An array of 3D covariance matrices */
typedef std::vector<cov3_mat_t, Eigen::aligned_allocator<cov3_mat_t> >
cov3_mats_t;

/** A 3-vector containing the cayley parameters of a rotation matrix */
typedef Eigen::Vector3d
    cayley_t;

/** A 4-vector containing the quaternion parameters of rotation matrix */
typedef Eigen::Vector4d
    quaternion_t;

/** Essential matrix \f$ \mathbf{E} \f$ between two viewpoints:
 *
 *  \f$ \mathbf{E} = \f$ skew(\f$\mathbf{t}\f$) \f$ \mathbf{R} \f$,
 *
 *  where \f$ \mathbf{t} \f$ describes the position of viewpoint 2 seen from
 *  viewpoint 1, and \f$\mathbf{R}\f$ describes the rotation from viewpoint 2
 *  to viewpoint 1.
 */
typedef Eigen::Matrix3d
    essential_t;

/** An array of essential matrices */
typedef std::vector<essential_t, Eigen::aligned_allocator<essential_t> >
    essentials_t;

/** An essential matrix with complex entires (as returned from
 *  fivept_stewenius [5])
 */
typedef Eigen::Matrix3cd
    complexEssential_t;

/** An array of complex-type essential matrices */
typedef std::vector< complexEssential_t, Eigen::aligned_allocator< complexEssential_t> >
    complexEssentials_t;

/** A 3-vector describing a point in 3D-space */
typedef Eigen::Vector3d
    point_t;

/** An array of 3D-points */
typedef std::vector<point_t, Eigen::aligned_allocator<point_t> >
    points_t;

/** A 3-vector containing the Eigenvalues of matrix \f$ \mathbf{M} \f$ in the
 *  eigensolver-algorithm (described in [11])
 */
typedef Eigen::Vector3d
    eigenvalues_t;

/** A 3x3 matrix containing the eigenvectors of matrix \f$ \mathbf{M} \f$ in the
 *  eigensolver-algorithm (described in [11])
 */
typedef Eigen::Matrix3d
    eigenvectors_t;

/** EigensolverOutput holds the output-parameters of the eigensolver-algorithm
 *  (described in [11])
 */
typedef struct EigensolverOutput
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** Position of viewpoint 2 seen from viewpoint 1 (unscaled) */
  translation_t   translation;
  /** Rotation from viewpoint 2 back to viewpoint 1 */
  rotation_t      rotation;
  /** The eigenvalues of matrix \f$ \mathbf{M} \f$ */
  eigenvalues_t   eigenvalues;
  /** The eigenvectors of matrix matrix \f$ \mathbf{M} \f$ */
  eigenvectors_t  eigenvectors;
} eigensolverOutput_t;

/** GeOutput holds the output-parameters of ge
 */
typedef struct GeOutput
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /** Homogeneous position of viewpoint 2 seen from viewpoint 1 */
  Eigen::Vector4d   translation;
  /** Rotation from viewpoint 2 back to viewpoint 1 */
  rotation_t        rotation;
  /** The eigenvalues of matrix \f$ \mathbf{G} \f$ */
  Eigen::Vector4d   eigenvalues;
  /** The eigenvectors of matrix matrix \f$ \mathbf{G} \f$ */
  Eigen::Matrix4d   eigenvectors;
} geOutput_t;


void mlpnp(
	const Eigen::Ref<const MatrixX3d>& object_points,
	const Eigen::Ref<const MatrixX3d>& proj_ray,
	const Eigen::Ref<const Eigen::Matrix3d>& convMat,
    const bool use_cov,
    Eigen::Vector3d& r_vec_out,
    Eigen::Vector3d& t_vec_out);