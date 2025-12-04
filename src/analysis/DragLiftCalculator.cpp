#include "DragLiftCalculator.hpp"

#include <algorithm>
#include <cmath>

#include "../core/D2Q9.hpp"

namespace lbm {

DragLiftCalculator::DragLiftCalculator(const SimulationConfig& config)
    : config_(config) {
    // Set reference velocity and length based on configuration
    if (!config_.obstacles.empty()) {
        reference_velocity_ = 0.1;  // Inflow velocity for cylinder flow
        // Find cylinder obstacle to get diameter
        for (const auto& obstacle : config_.obstacles) {
            if (obstacle.type == "cylinder" && obstacle.parameters.size() >= 3) {
                reference_length_ = 2.0 * obstacle.parameters[2];  // diameter = 2 * radius
                break;
            }
        }
    } else {
        reference_velocity_ = config_.lid_velocity;
        reference_length_ = static_cast<double>(config_.ny);  // Channel height for cavity
    }
}

ForceResult DragLiftCalculator::compute_forces_cpu(
    const std::vector<double>& f_curr,
    const std::vector<bool>& obstacle_mask) const {
    return compute_forces_impl(
        f_curr.data(),
        obstacle_mask,
        config_.nx,
        config_.ny);
}

ForceResult DragLiftCalculator::compute_forces_cuda(
    const std::vector<double>& f_curr,
    const std::vector<bool>& obstacle_mask) const {
    return compute_forces_impl(
        f_curr.data(),
        obstacle_mask,
        config_.nx,
        config_.ny);
}

ForceResult DragLiftCalculator::compute_forces_impl(
    const double* f_curr,
    const std::vector<bool>& obstacle_mask,
    std::size_t nx,
    std::size_t ny) const {
    
    ForceResult result;
    double drag_force = 0.0;
    double lift_force = 0.0;
    
    // Momentum exchange method: compute forces on obstacle surface
    // For each obstacle cell, sum momentum exchange with neighboring fluid cells
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const std::size_t cell = y * nx + x;
            
            if (cell >= obstacle_mask.size() || !obstacle_mask[cell]) {
                continue;  // Skip non-obstacle cells
            }
            
            // Check all neighbors - if neighbor is fluid, compute momentum exchange
            for (int q = 0; q < D2Q9::q; ++q) {
                const int xn = static_cast<int>(x) + D2Q9::cx[q];
                const int yn = static_cast<int>(y) + D2Q9::cy[q];
                
                // Check if neighbor is within bounds and is a fluid cell
                if (xn >= 0 && yn >= 0 && xn < static_cast<int>(nx) && yn < static_cast<int>(ny)) {
                    const std::size_t neighbor_cell = static_cast<std::size_t>(yn) * nx + static_cast<std::size_t>(xn);
                    
                    if (neighbor_cell >= obstacle_mask.size() || !obstacle_mask[neighbor_cell]) {
                        // Neighbor is fluid - compute momentum exchange
                        // The distribution function f[q] at the fluid cell contributes to force
                        const std::size_t neighbor_base = neighbor_cell * D2Q9::q;
                        const double f_val = f_curr[neighbor_base + q];
                        
                        // Momentum exchange: force = 2 * f * c (for bounce-back)
                        // Direction q points from fluid to obstacle
                        const double fx = 2.0 * f_val * static_cast<double>(D2Q9::cx[q]);
                        const double fy = 2.0 * f_val * static_cast<double>(D2Q9::cy[q]);
                        
                        drag_force += fx;
                        lift_force += fy;
                    }
                }
            }
        }
    }
    
    result.drag_force = drag_force;
    result.lift_force = lift_force;
    
    // Compute coefficients: Cd = Fd / (0.5 * rho * U^2 * L) for 2D
    // In LBM, rho ≈ 1.0, so: Cd = 2 * Fd / (U^2 * L)
    const double rho_ref = 1.0;
    const double denom = 0.5 * rho_ref * reference_velocity_ * reference_velocity_ * reference_length_;
    
    if (std::abs(denom) > 1e-10) {
        result.drag_coefficient = 2.0 * drag_force / (reference_velocity_ * reference_velocity_ * reference_length_);
        result.lift_coefficient = 2.0 * lift_force / (reference_velocity_ * reference_velocity_ * reference_length_);
    }
    
    return result;
}

}  // namespace lbm

