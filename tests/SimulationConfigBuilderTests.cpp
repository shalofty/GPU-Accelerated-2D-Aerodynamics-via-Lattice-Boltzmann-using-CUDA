#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/io/SimulationConfigBuilder.hpp"

TEST_CASE("SimulationConfigBuilder produces default configuration", "[config]") {
    lbm::SimulationConfigBuilder builder;
    builder.set_config_path("dummy.yaml");
    auto config = builder.build();

    REQUIRE(config.nx == 256);
    REQUIRE(config.ny == 256);
    REQUIRE(config.relaxation_time == Catch::Approx(0.6));
    REQUIRE(config.max_timesteps == 1000);
    REQUIRE(config.output_interval == 100);
    REQUIRE(config.lid_velocity == Catch::Approx(0.1));
    REQUIRE(config.residual_tolerance == Catch::Approx(1e-6));
}

TEST_CASE("SimulationConfigBuilder throws when path missing", "[config]") {
    lbm::SimulationConfigBuilder builder;
    REQUIRE_THROWS(builder.build());
}
