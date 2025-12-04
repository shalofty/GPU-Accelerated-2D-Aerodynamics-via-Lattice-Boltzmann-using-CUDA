#pragma once

#include <cstddef>
#include <vector>

#include "../core/SimulationConfig.hpp"

namespace lbm {

struct ForceResult {
    double drag_force{0.0};
    double lift_force{0.0};
    double drag_coefficient{0.0};
    double lift_coefficient{0.0};
};

class DragLiftCalculator {
  public:
    DragLiftCalculator(const SimulationConfig& config);
    
    // Compute forces from CPU backend data
    ForceResult compute_forces_cpu(
        const std::vector<double>& f_curr,
        const std::vector<bool>& obstacle_mask) const;
    
    // Compute forces from CUDA backend data (after copying to host)
    ForceResult compute_forces_cuda(
        const std::vector<double>& f_curr,
        const std::vector<bool>& obstacle_mask) const;

  private:
    ForceResult compute_forces_impl(
        const double* f_curr,
        const std::vector<bool>& obstacle_mask,
        std::size_t nx,
        std::size_t ny) const;
    
    SimulationConfig config_;
    double reference_velocity_{0.1};  // U_inflow or U_lid
    double reference_length_{1.0};     // Characteristic length (cylinder diameter)
};

}  // namespace lbm

