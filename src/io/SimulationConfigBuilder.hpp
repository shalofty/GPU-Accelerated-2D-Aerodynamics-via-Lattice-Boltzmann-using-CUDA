#pragma once

#include <filesystem>

#include "../core/SimulationConfig.hpp"

namespace lbm {

class SimulationConfigBuilder {
  public:
    SimulationConfigBuilder& set_config_path(std::filesystem::path path);
    SimulationConfig build() const;

  private:
    std::filesystem::path config_path_;
};

}  // namespace lbm
