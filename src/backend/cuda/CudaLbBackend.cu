#include "CudaLbBackend.hpp"

#include <stdexcept>

namespace lbm {

namespace {
__global__ void noop_kernel(double* /*data*/, std::size_t /*count*/) {}
}

void CudaLbBackend::initialize(const SimulationConfig& config) {
    config_ = config;
    timestep_ = 0;
    release_device_buffers();
    allocate_device_buffers(config);
}

void CudaLbBackend::step() {
    if (!device_distributions_) {
        return;
    }

    const std::size_t count = config_.nx * config_.ny * 9;  // D2Q9 populations
    const dim3 block_dim(256);
    const dim3 grid_dim((count + block_dim.x - 1) / block_dim.x);
    noop_kernel<<<grid_dim, block_dim>>>(device_distributions_, count);
    cudaDeviceSynchronize();

    if (timestep_ < config_.max_timesteps) {
        ++timestep_;
    }
}

bool CudaLbBackend::is_converged() const {
    return timestep_ >= config_.max_timesteps;
}

std::size_t CudaLbBackend::current_timestep() const {
    return timestep_;
}

DiagnosticSnapshot CudaLbBackend::fetch_diagnostics() const {
    DiagnosticSnapshot snapshot{};
    snapshot.timestep = timestep_;
    return snapshot;
}

void CudaLbBackend::allocate_device_buffers(const SimulationConfig& config) {
    const std::size_t count = config.nx * config.ny * 9;
    const std::size_t bytes = count * sizeof(double);
    if (cudaMalloc(&device_distributions_, bytes) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device distributions");
    }
}

void CudaLbBackend::release_device_buffers() {
    if (device_distributions_) {
        cudaFree(device_distributions_);
        device_distributions_ = nullptr;
    }
}

CudaLbBackend::~CudaLbBackend() {
    release_device_buffers();
}

}  // namespace lbm
