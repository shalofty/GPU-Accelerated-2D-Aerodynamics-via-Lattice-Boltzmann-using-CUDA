#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "DiagnosticObserver.hpp"
#include "../io/VtkWriter.hpp"
#include "../core/SimulationBackend.hpp"

namespace lbm {

class VtkObserver final : public DiagnosticObserver {
  public:
    VtkObserver(
        std::filesystem::path output_dir,
        std::size_t output_interval,
        SimulationBackend* backend);
    
    void on_step(const DiagnosticSnapshot& snapshot) override;

  private:
    std::filesystem::path output_dir_;
    std::size_t output_interval_;
    SimulationBackend* backend_;
    VtkWriter writer_;
    
    void write_fields_from_cpu_backend(const DiagnosticSnapshot& snapshot);
    void write_fields_from_cuda_backend(const DiagnosticSnapshot& snapshot);
};

}  // namespace lbm

