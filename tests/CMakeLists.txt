# Fetch Catch2 if not available
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(lbm_tests
    SimulationConfigBuilderTests.cpp
    CpuLbBackendTests.cpp
    CpuCudaComparisonTests.cpp)

target_link_libraries(lbm_tests
    PRIVATE
        lbm_core
        Catch2::Catch2WithMain)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(lbm_tests PRIVATE lbm_cuda)
    target_compile_definitions(lbm_tests PRIVATE ENABLE_CUDA)
endif()

target_include_directories(lbm_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(CTest)
include(Catch)
catch_discover_tests(lbm_tests)
