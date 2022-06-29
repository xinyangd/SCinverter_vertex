add_library(catch2 catch2/catch2.cpp)

foreach(f utils k_space_structure single_freq_gf F)
    add_executable(${f} ${f}.cpp)

    target_link_libraries(${f} catch2 SCinverter)

    target_include_directories(${f} PRIVATE catch2)

    target_compile_definitions(${f} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/test_files/")

    add_test(NAME ${f}_test COMMAND ${f} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/test_files/)
endforeach()

include_directories(../src)
include_directories(../src/lib)
include_directories(../src/task)
include_directories(../src/task/freq_meas)
include_directories(./)