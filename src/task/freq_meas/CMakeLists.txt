include_directories("./")
include_directories("../..")

add_executable(cluster_susc_task cluster_susc_main.cpp cluster_susc_task.cpp)
target_link_libraries(cluster_susc_task SCinverter)

add_executable(full_vertex_task full_vertex_main.cpp full_vertex_task.cpp cluster_susc_task.cpp)
target_link_libraries(full_vertex_task SCinverter)

add_executable(fluctuation_task fluctuation_main.cpp fluctuation_task.cpp full_vertex_task.cpp cluster_susc_task.cpp)
target_link_libraries(fluctuation_task SCinverter)