#pragma once

#include <cstddef>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#else
// Forward declarations for when CUDA is not available
typedef void* cudaStream_t;
#endif

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
    
    // Field data accessors for visualization
    void get_field_data(std::vector<double>& density, std::vector<double>& ux, std::vector<double>& uy) const;
    void get_distributions(std::vector<double>& f_curr) const;
    void get_obstacle_mask(std::vector<bool>& obstacle_mask) const;
    const SimulationConfig& config() const { return config_; }

  private:
    void allocate_device_buffers(const SimulationConfig& config);
    void release_device_buffers();
    void initialize_lattice();
    void build_obstacle_mask();
    void collide_and_stream();
    void collide_and_stream_standard();
    void collide_and_stream_tiled();
    void apply_boundary_conditions();
    void apply_lid_velocity();
    void apply_inflow_outflow();
    void compute_macro_fields();
    
    bool use_tiled_kernel_{true};  // Use shared-memory tiled kernel by default

    SimulationConfig config_;
    std::size_t timestep_{0};
    
    // Double buffering for distributions
    double* f_curr_{nullptr};
    double* f_next_{nullptr};
    
    // Macroscopic fields
    double* density_{nullptr};
    double* ux_{nullptr};
    double* uy_{nullptr};
    
    // Obstacle mask
    bool* obstacle_mask_{nullptr};
    
    // CUDA streams for async operations
    cudaStream_t compute_stream_{nullptr};
    cudaStream_t transfer_stream_{nullptr};
    
    // Residual (computed on host)
    double residual_{0.0};
};

}  // namespace lbm
