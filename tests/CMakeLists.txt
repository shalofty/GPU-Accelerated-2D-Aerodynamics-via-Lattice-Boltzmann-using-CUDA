add_subdirectory(../external/Catch2 Catch2-build)

add_executable(lbm_tests
    SimulationConfigBuilderTests.cpp
    CpuLbBackendTests.cpp)

target_link_libraries(lbm_tests
    PRIVATE
        lbm_core
        Catch2::Catch2WithMain)

target_include_directories(lbm_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(CTest)
include(Catch)
catch_discover_tests(lbm_tests)
