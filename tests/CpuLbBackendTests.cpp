#include <catch2/catch_test_macros.hpp>

#include "../src/backend/cpu/CpuLbBackend.hpp"

namespace {
lbm::SimulationConfig make_config() {
    lbm::SimulationConfig config{};
    config.nx = 16;
    config.ny = 16;
    config.relaxation_time = 0.6;
    config.max_timesteps = 10;
    config.output_interval = 5;
    config.lid_velocity = 0.1;
    config.residual_tolerance = 1e-3;
    config.backend_id = "cpu";
    return config;
}
}  // namespace

TEST_CASE("CpuLbBackend initializes and steps", "[cpu]") {
    lbm::CpuLbBackend backend;
    auto config = make_config();

    backend.initialize(config);
    REQUIRE(backend.current_timestep() == 0);
    REQUIRE_FALSE(backend.is_converged());

    backend.step();
    REQUIRE(backend.current_timestep() == 1);

    auto snapshot = backend.fetch_diagnostics();
    REQUIRE(snapshot.timestep == 1);
}

TEST_CASE("CpuLbBackend reaches convergence", "[cpu]") {
    lbm::CpuLbBackend backend;
    auto config = make_config();
    config.max_timesteps = 2;

    backend.initialize(config);
    backend.step();
    backend.step();
    REQUIRE(backend.is_converged());
}
