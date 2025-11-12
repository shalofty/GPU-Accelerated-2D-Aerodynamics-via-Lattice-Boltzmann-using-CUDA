cmake_minimum_required(VERSION 3.22)

project(GPUAcceleratedLBM
    VERSION 0.1.0
    DESCRIPTION "GPU-Accelerated 2D Lattice Boltzmann Solver"
    LANGUAGES CXX CUDA)

option(ENABLE_CUDA "Enable CUDA backend" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_subdirectory(src)
add_subdirectory(tests)

install(DIRECTORY docs/ DESTINATION share/${PROJECT_NAME}/docs OPTIONAL)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts OPTIONAL)
