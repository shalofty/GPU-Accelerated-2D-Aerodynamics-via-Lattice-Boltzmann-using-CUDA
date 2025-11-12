#include "SimulationConfigBuilder.hpp"

#include <stdexcept>

namespace lbm {

SimulationConfigBuilder& SimulationConfigBuilder::set_config_path(std::filesystem::path path) {
    config_path_ = std::move(path);
    return *this;
}

SimulationConfig SimulationConfigBuilder::build() const {
    if (config_path_.empty()) {
        throw std::runtime_error("Configuration path not set");
    }

    // TODO: Parse YAML/TOML configuration file.
    SimulationConfig config{};
    config.nx = 256;
    config.ny = 256;
    config.relaxation_time = 0.6;
    config.max_timesteps = 1000;
    config.output_interval = 100;
    config.lid_velocity = 0.1;
    config.residual_tolerance = 1e-6;
    return config;
}

}  // namespace lbm
