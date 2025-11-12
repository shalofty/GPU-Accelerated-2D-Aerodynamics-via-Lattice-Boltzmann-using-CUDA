set(LBM_CORE_SOURCES
    core/SimulationRunner.cpp
    analysis/PerformanceLogger.cpp
    io/SimulationConfigBuilder.cpp
    backend/cpu/CpuLbBackend.cpp
)

add_library(lbm_core STATIC ${LBM_CORE_SOURCES})

target_include_directories(lbm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(lbm_core PUBLIC cxx_std_20)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_library(lbm_cuda STATIC backend/cuda/CudaLbBackend.cu)
    target_include_directories(lbm_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(lbm_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(lbm_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(lbm_cuda PUBLIC lbm_core)
else()
    message(STATUS "CUDA backend disabled or compiler not found; skipping lbm_cuda target")
endif()
