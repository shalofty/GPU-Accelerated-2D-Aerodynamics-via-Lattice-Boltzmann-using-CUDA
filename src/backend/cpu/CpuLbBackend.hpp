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
    
    // Field data accessors for visualization
    const std::vector<double>& density() const { return density_; }
    const std::vector<double>& ux() const { return ux_; }
    const std::vector<double>& uy() const { return uy_; }
    const std::vector<double>& f_curr() const { return f_curr_; }
    const std::vector<bool>& obstacle_mask() const { return obstacle_mask_; }
    const SimulationConfig& config() const { return config_; }

  private:
    std::size_t lattice_index(std::size_t x, std::size_t y, std::size_t q) const;
    std::size_t cell_index(std::size_t x, std::size_t y) const;
    void initialize_lattice();
    void build_obstacle_mask();
    bool is_obstacle_cell(std::size_t x, std::size_t y) const;
    bool is_inside_cylinder(double x, double y, double cx, double cy, double radius) const;
    void collide_and_stream();
    void apply_boundary_conditions();
    void apply_lid_velocity();
    void apply_inflow_outflow();
    void apply_obstacle_bounce_back();
    void compute_macro_fields();

    SimulationConfig config_;
    std::size_t timestep_{0};
    std::vector<double> f_curr_;
    std::vector<double> f_next_;
    std::vector<double> density_;
    std::vector<double> ux_;
    std::vector<double> uy_;
    std::vector<bool> obstacle_mask_;
    double residual_{0.0};
};

}  // namespace lbm
