#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "../core/SimulationBackend.hpp"
#include "../core/SimulationConfig.hpp"
#include "DragLiftCalculator.hpp"

namespace lbm {

struct ValidationResult {
    std::string test_name;
    bool passed{false};
    double expected_value{0.0};
    double computed_value{0.0};
    double relative_error{0.0};
    double tolerance{0.05};  // 5% default tolerance
    std::string message;
};

class ValidationSuite {
  public:
    ValidationSuite();
    
    // Run validation tests
    std::vector<ValidationResult> run_validation(
        SimulationBackend* backend,
        const std::string& test_case);
    
    // Generate golden data (reference solutions)
    void generate_golden_data(
        SimulationBackend* backend,
        const std::filesystem::path& output_dir);
    
    // Compare results against golden data
    std::vector<ValidationResult> compare_with_golden(
        SimulationBackend* backend,
        const std::filesystem::path& golden_data_dir);

  private:
    ValidationResult validate_lid_driven_cavity(SimulationBackend* backend);
    ValidationResult validate_cylinder_flow(SimulationBackend* backend);
    
    // Reference values (from literature)
    struct ReferenceValues {
        // Lid-driven cavity at Re=1000, center velocity
        static constexpr double cavity_center_ux_ref = 0.0;  // Approximate
        static constexpr double cavity_center_uy_ref = 0.0;  // Approximate
        
        // Cylinder flow at Re=20, drag coefficient
        static constexpr double cylinder_cd_ref = 2.0;  // Approximate for Re=20
        static constexpr double cylinder_cl_ref = 0.0;  // Symmetric case
    };
};

}  // namespace lbm

