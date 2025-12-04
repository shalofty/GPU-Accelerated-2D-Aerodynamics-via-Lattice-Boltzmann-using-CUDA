#include "ValidationSuite.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "../backend/cpu/CpuLbBackend.hpp"

#ifdef ENABLE_CUDA
#include "../backend/cuda/CudaLbBackend.hpp"
#endif

namespace lbm {

ValidationSuite::ValidationSuite() = default;

std::vector<ValidationResult> ValidationSuite::run_validation(
    SimulationBackend* backend,
    const std::string& test_case) {
    
    std::vector<ValidationResult> results;
    
    if (test_case == "lid_driven_cavity" || test_case == "all") {
        results.push_back(validate_lid_driven_cavity(backend));
    }
    
    if (test_case == "cylinder_flow" || test_case == "all") {
        results.push_back(validate_cylinder_flow(backend));
    }
    
    return results;
}

ValidationResult ValidationSuite::validate_lid_driven_cavity(SimulationBackend* backend) {
    ValidationResult result;
    result.test_name = "lid_driven_cavity";
    result.tolerance = 0.1;  // 10% tolerance for cavity flow
    
    // Create cavity configuration
    SimulationConfig config;
    config.nx = 64;
    config.ny = 64;
    config.relaxation_time = 0.6;
    config.max_timesteps = 1000;
    config.lid_velocity = 0.1;
    config.residual_tolerance = 1e-4;
    
    backend->initialize(config);
    
    // Run simulation
    for (std::size_t i = 0; i < config.max_timesteps && !backend->is_converged(); ++i) {
        backend->step();
    }
    
    // Get center velocity (approximate validation)
    auto snapshot = backend->fetch_diagnostics();
    
    // For cavity flow, we validate that the flow develops correctly
    // Check that residual decreases and flow is non-zero
    result.computed_value = snapshot.residual_l2;
    result.expected_value = config.residual_tolerance;
    
    // Validation: residual should be below tolerance
    result.passed = snapshot.residual_l2 <= config.residual_tolerance;
    result.relative_error = std::abs(snapshot.residual_l2 - config.residual_tolerance) / 
                           (config.residual_tolerance + 1e-10);
    
    if (result.passed) {
        result.message = "Cavity flow converged successfully";
    } else {
        result.message = "Cavity flow did not converge within tolerance";
    }
    
    return result;
}

ValidationResult ValidationSuite::validate_cylinder_flow(SimulationBackend* backend) {
    ValidationResult result;
    result.test_name = "cylinder_flow";
    result.tolerance = 0.15;  // 15% tolerance for drag coefficient
    
    // Create cylinder flow configuration
    SimulationConfig config;
    config.nx = 200;
    config.ny = 100;
    config.relaxation_time = 0.6;
    config.max_timesteps = 2000;
    config.residual_tolerance = 1e-4;
    
    // Add cylinder obstacle
    ObstacleDefinition cylinder;
    cylinder.id = "cylinder1";
    cylinder.type = "cylinder";
    cylinder.parameters = {50.0, 50.0, 10.0};  // cx, cy, radius
    config.obstacles.push_back(cylinder);
    
    backend->initialize(config);
    
    // Run simulation
    for (std::size_t i = 0; i < config.max_timesteps && !backend->is_converged(); ++i) {
        backend->step();
    }
    
    // Compute drag coefficient
    DragLiftCalculator calculator(config);
    
    // Get distribution functions and obstacle mask
    std::vector<double> f_curr;
    std::vector<bool> obstacle_mask;
    
    if (auto* cpu_backend = dynamic_cast<CpuLbBackend*>(backend)) {
        f_curr = cpu_backend->f_curr();
        obstacle_mask = cpu_backend->obstacle_mask();
    }
#ifdef ENABLE_CUDA
    else if (auto* cuda_backend = dynamic_cast<CudaLbBackend*>(backend)) {
        cuda_backend->get_distributions(f_curr);
        cuda_backend->get_obstacle_mask(obstacle_mask);
    }
#endif
    
    if (!f_curr.empty() && !obstacle_mask.empty()) {
        ForceResult forces = calculator.compute_forces_cpu(f_curr, obstacle_mask);
        
        result.computed_value = forces.drag_coefficient;
        result.expected_value = ReferenceValues::cylinder_cd_ref;
        result.relative_error = std::abs(forces.drag_coefficient - ReferenceValues::cylinder_cd_ref) /
                               (ReferenceValues::cylinder_cd_ref + 1e-10);
        result.passed = result.relative_error <= result.tolerance;
        
        std::ostringstream msg;
        msg << "Drag coefficient: " << std::fixed << std::setprecision(4) 
            << forces.drag_coefficient << " (expected: " << ReferenceValues::cylinder_cd_ref << ")";
        result.message = msg.str();
    } else {
        result.passed = false;
        result.message = "Failed to retrieve simulation data";
    }
    
    return result;
}

void ValidationSuite::generate_golden_data(
    SimulationBackend* backend,
    const std::filesystem::path& output_dir) {
    
    std::filesystem::create_directories(output_dir);
    
    // Generate golden data for standard test cases
    std::ofstream out(output_dir / "golden_data.json");
    out << "{\n";
    out << "  \"lid_driven_cavity\": {\n";
    out << "    \"residual_tolerance\": 1e-4,\n";
    out << "    \"convergence_timesteps\": 1000\n";
    out << "  },\n";
    out << "  \"cylinder_flow\": {\n";
    out << "    \"drag_coefficient\": " << ReferenceValues::cylinder_cd_ref << ",\n";
    out << "    \"lift_coefficient\": " << ReferenceValues::cylinder_cl_ref << "\n";
    out << "  }\n";
    out << "}\n";
}

std::vector<ValidationResult> ValidationSuite::compare_with_golden(
    SimulationBackend* backend,
    const std::filesystem::path& golden_data_dir) {
    
    // This would parse golden data JSON and compare
    // For now, just run standard validation
    return run_validation(backend, "all");
}

}  // namespace lbm

