#pragma once

#include <cstddef>
#include <vector>

#include "../../core/SimulationBackend.hpp"

namespace lbm {

class CpuLbBackend final : public SimulationBackend {
  public:
    void initialize(const SimulationConfig& config) override;
    void step() override;
    bool is_converged() const override;
    std::size_t current_timestep() const override;
    DiagnosticSnapshot fetch_diagnostics() const override;

  private:
    std::size_t lattice_index(std::size_t x, std::size_t y, std::size_t q) const;
    std::size_t cell_index(std::size_t x, std::size_t y) const;
    void initialize_lattice();
    void collide_and_stream();
    void apply_lid_velocity();
    void compute_macro_fields();

    SimulationConfig config_;
    std::size_t timestep_{0};
    std::vector<double> f_curr_;
    std::vector<double> f_next_;
    std::vector<double> density_;
    std::vector<double> ux_;
    std::vector<double> uy_;
    double residual_{0.0};
};

}  // namespace lbm
