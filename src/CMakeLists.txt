add_library(SCinverter
        lib/parameters.cpp
        lib/k_space_structure.cpp
        lib/single_freq_gf.cpp
        lib/mc_vertex.cpp
        lib/vertex.cpp
        lib/chi0_phpp.cpp
        lib/chi_phpp.cpp
        lib/F.cpp
        )
target_link_libraries(SCinverter PUBLIC
        ${CLUSTER_LIBRARIES}
        ${ALPSCore_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        ${LAPACK_LIBRARIES})

include_directories("lib")
include_directories("task")
include_directories("./")

set(t "SCinverter")

if(TARGET Eigen3::Eigen)
    target_link_libraries(${t} PRIVATE Eigen3::Eigen)
else(TARGET Eigen3::Eigen)
    target_compile_definitions(${t} PRIVATE ${EIGEN3_DEFINITIONS})
    target_include_directories(${t} PUBLIC ${EIGEN3_INCLUDE_DIRS})
endif(TARGET Eigen3::Eigen)

add_subdirectory(task)

