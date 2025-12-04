#include "VtkObserver.hpp"

#include <iomanip>
#include <sstream>
#include <vector>

#include "../backend/cpu/CpuLbBackend.hpp"

#ifdef ENABLE_CUDA
#include "../backend/cuda/CudaLbBackend.hpp"
#include <cuda_runtime.h>
#endif

namespace lbm {

VtkObserver::VtkObserver(
    std::filesystem::path output_dir,
    std::size_t output_interval,
    SimulationBackend* backend)
    : output_dir_(std::move(output_dir))
    , output_interval_(output_interval)
    , backend_(backend)
    , writer_(output_dir_) {
}

void VtkObserver::on_step(const DiagnosticSnapshot& snapshot) {
    if (snapshot.timestep % output_interval_ != 0) {
        return;
    }
    
    // Try CPU backend first
    if (auto* cpu_backend = dynamic_cast<CpuLbBackend*>(backend_)) {
        write_fields_from_cpu_backend(snapshot);
        return;
    }
    
#ifdef ENABLE_CUDA
    // Try CUDA backend
    if (auto* cuda_backend = dynamic_cast<CudaLbBackend*>(backend_)) {
        write_fields_from_cuda_backend(snapshot);
        return;
    }
#endif
}

void VtkObserver::write_fields_from_cpu_backend(const DiagnosticSnapshot& snapshot) {
    auto* cpu_backend = dynamic_cast<CpuLbBackend*>(backend_);
    if (!cpu_backend) {
        return;
    }
    
    const auto& density = cpu_backend->density();
    const auto& ux = cpu_backend->ux();
    const auto& uy = cpu_backend->uy();
    const auto& config = cpu_backend->config();
    
    std::ostringstream filename;
    filename << "field_" << std::setfill('0') << std::setw(6) << snapshot.timestep << ".vtk";
    
    writer_.write_field(
        filename.str(),
        config.nx,
        config.ny,
        density.data(),
        ux.data(),
        uy.data(),
        snapshot.timestep);
}

void VtkObserver::write_fields_from_cuda_backend(const DiagnosticSnapshot& snapshot) {
#ifdef ENABLE_CUDA
    auto* cuda_backend = dynamic_cast<CudaLbBackend*>(backend_);
    if (!cuda_backend) {
        return;
    }
    
    std::vector<double> density, ux, uy;
    cuda_backend->get_field_data(density, ux, uy);
    
    // Check if data was retrieved successfully
    if (density.empty() || ux.empty() || uy.empty()) {
        return;  // Skip if data not available
    }
    
    const auto& config = cuda_backend->config();
    
    std::ostringstream filename;
    filename << "field_" << std::setfill('0') << std::setw(6) << snapshot.timestep << ".vtk";
    
    writer_.write_field(
        filename.str(),
        config.nx,
        config.ny,
        density.data(),
        ux.data(),
        uy.data(),
        snapshot.timestep);
#endif
}

}  // namespace lbm

