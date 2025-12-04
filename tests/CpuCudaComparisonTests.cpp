#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../src/backend/cpu/CpuLbBackend.hpp"

#ifdef ENABLE_CUDA
#include "../src/backend/cuda/CudaLbBackend.hpp"
#endif

#include <cmath>

namespace {
lbm::SimulationConfig make_small_config() {
    lbm::SimulationConfig config{};
    config.nx = 32;
    config.ny = 32;
    config.relaxation_time = 0.6;
    config.max_timesteps = 50;
    config.output_interval = 10;
    config.lid_velocity = 0.1;
    config.residual_tolerance = 1e-6;
    return config;
}
}  // namespace

#ifdef ENABLE_CUDA

TEST_CASE("CPU and CUDA backends produce similar results", "[comparison][cuda]") {
    auto config = make_small_config();
    
    lbm::CpuLbBackend cpu_backend;
    lbm::CudaLbBackend cuda_backend;
    
    cpu_backend.initialize(config);
    cuda_backend.initialize(config);
    
    // Run a few steps
    for (int i = 0; i < 10; ++i) {
        cpu_backend.step();
        cuda_backend.step();
    }
    
    // Compare diagnostics
    auto cpu_snapshot = cpu_backend.fetch_diagnostics();
    auto cuda_snapshot = cuda_backend.fetch_diagnostics();
    
    REQUIRE(cpu_snapshot.timestep == cuda_snapshot.timestep);
    
    // Residuals should be similar (within 1% relative tolerance)
    const double residual_tol = std::max(1e-6, std::abs(cpu_snapshot.residual_l2) * 0.01);
    REQUIRE(cuda_snapshot.residual_l2 == Catch::Approx(cpu_snapshot.residual_l2).margin(residual_tol));
    
    // Drag/lift coefficients should be similar
    const double coeff_tol = 0.01;  // 1% tolerance
    REQUIRE(cuda_snapshot.drag_coefficient == Catch::Approx(cpu_snapshot.drag_coefficient).margin(coeff_tol));
    REQUIRE(cuda_snapshot.lift_coefficient == Catch::Approx(cpu_snapshot.lift_coefficient).margin(coeff_tol));
}

TEST_CASE("CPU and CUDA backends converge similarly", "[comparison][cuda]") {
    auto config = make_small_config();
    config.max_timesteps = 100;
    config.residual_tolerance = 1e-4;
    
    lbm::CpuLbBackend cpu_backend;
    lbm::CudaLbBackend cuda_backend;
    
    cpu_backend.initialize(config);
    cuda_backend.initialize(config);
    
    // Run until convergence or max steps
    while (!cpu_backend.is_converged() && !cuda_backend.is_converged()) {
        cpu_backend.step();
        cuda_backend.step();
        
        // Check they're at same timestep
        REQUIRE(cpu_backend.current_timestep() == cuda_backend.current_timestep());
    }
    
    // Both should converge around the same time (within a few steps)
    const auto cpu_timestep = cpu_backend.current_timestep();
    const auto cuda_timestep = cuda_backend.current_timestep();
    REQUIRE(std::abs(static_cast<int>(cpu_timestep) - static_cast<int>(cuda_timestep)) <= 5);
    
    // Final residuals should be similar
    auto cpu_snapshot = cpu_backend.fetch_diagnostics();
    auto cuda_snapshot = cuda_backend.fetch_diagnostics();
    
    const double residual_tol = std::max(1e-6, std::abs(cpu_snapshot.residual_l2) * 0.05);
    REQUIRE(cuda_snapshot.residual_l2 == Catch::Approx(cpu_snapshot.residual_l2).margin(residual_tol));
}

#else

TEST_CASE("CUDA backend not available (ENABLE_CUDA=OFF)", "[comparison]") {
    // This test always passes when CUDA is disabled
    REQUIRE(true);
}

#endif

