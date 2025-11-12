#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace lbm {

struct ObstacleDefinition {
    std::string id;
    std::string type; // e.g., "cylinder", "rectangle"
    std::vector<double> parameters;
};

struct SimulationConfig {
    std::size_t nx{};
    std::size_t ny{};
    double relaxation_time{};
    std::size_t max_timesteps{};
    std::size_t output_interval{};
    double lid_velocity{0.1};
    double residual_tolerance{1e-6};
    std::vector<ObstacleDefinition> obstacles{};
    std::string backend_id{"cuda"};
};

}  // namespace lbm
