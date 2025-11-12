#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "../../core/SimulationBackend.hpp"

namespace lbm {

class CudaLbBackend final : public SimulationBackend {
  public:
    ~CudaLbBackend() override;
    void initialize(const SimulationConfig& config) override;
    void step() override;
    bool is_converged() const override;
    std::size_t current_timestep() const override;
    DiagnosticSnapshot fetch_diagnostics() const override;

  private:
    void allocate_device_buffers(const SimulationConfig& config);
    void release_device_buffers();

    SimulationConfig config_;
    std::size_t timestep_{0};
    double* device_distributions_{nullptr};
};

}  // namespace lbm
