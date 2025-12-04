#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <cstdio>

#include "../src/io/SimulationConfigBuilder.hpp"

TEST_CASE("SimulationConfigBuilder produces default configuration", "[config]") {
    // Create a temporary config file
    std::ofstream test_file("test_config.yaml");
    test_file << "# Test config file\n";
    test_file.close();
    
    lbm::SimulationConfigBuilder builder;
    builder.set_config_path("test_config.yaml");
    auto config = builder.build();

    REQUIRE(config.nx == 256);
    REQUIRE(config.ny == 256);
    REQUIRE(config.relaxation_time == Catch::Approx(0.6));
    REQUIRE(config.max_timesteps == 1000);
    REQUIRE(config.output_interval == 100);
    REQUIRE(config.lid_velocity == Catch::Approx(0.1));
    REQUIRE(config.residual_tolerance == Catch::Approx(1e-6));
    
    // Clean up
    std::remove("test_config.yaml");
}

TEST_CASE("SimulationConfigBuilder throws when path missing", "[config]") {
    lbm::SimulationConfigBuilder builder;
    REQUIRE_THROWS(builder.build());
}
