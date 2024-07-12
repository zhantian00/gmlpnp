cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(CMAKE_BUILD_TYPE Release)

project(advancedpnp)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11/)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/g2o/cmake_modules)


set(opengv_DIR ${PROJECT_SOURCE_DIR}/thirdparty/opengv/build)
find_package(opengv REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
)

pybind11_add_module(${PROJECT_NAME}
src/binding.cpp
src/ppnp.cpp 
src/dlt.cpp 
src/opengv_wrapper.cpp
src/mlpnp.cpp 
src/cpnp.cpp
src/gmlpnp.cpp
src/gmlpnp_with_prior.cpp 
)



target_link_libraries(${PROJECT_NAME} PRIVATE
g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_solver_dense g2o_types_slam3d opengv Eigen3::Eigen
)