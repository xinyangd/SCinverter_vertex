cmake_minimum_required(VERSION 3.1)
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()
project(SCinv_vert LANGUAGES CXX)
 
set(CMAKE_CXX_STANDARD 11)

find_package(ALPSCore REQUIRED COMPONENTS hdf5 gf params)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DMFT_DIR}/cmake")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(BlasClasses REQUIRED)
find_package(Cluster REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)

include_directories(${BLAS_CLASSES_INCLUDE_DIR})
include_directories(${CLUSTER_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -DBOOST_DISABLE_ASSERTS -DALPS_GF_DEBUG=NDEBUG")

add_subdirectory(src)

option(Testing "enable testing" ON)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif()
