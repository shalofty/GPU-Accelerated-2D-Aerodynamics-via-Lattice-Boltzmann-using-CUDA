#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "SimulationConfig.hpp"

namespace lbm {

struct DiagnosticSnapshot {
    std::size_t timestep{};
    double residual_l2{};
    double lift_coefficient{};
    double drag_coefficient{};
};

class SimulationBackend {
  public:
    virtual ~SimulationBackend() = default;

    virtual void initialize(const SimulationConfig& config) = 0;
    virtual void step() = 0;
    virtual bool is_converged() const = 0;
    virtual std::size_t current_timestep() const = 0;
    virtual DiagnosticSnapshot fetch_diagnostics() const = 0;
};

using SimulationBackendPtr = std::unique_ptr<SimulationBackend>;

}  // namespace lbm
