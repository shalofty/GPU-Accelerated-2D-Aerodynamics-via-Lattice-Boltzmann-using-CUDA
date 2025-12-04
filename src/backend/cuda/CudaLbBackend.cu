#include "CudaLbBackend.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../core/D2Q9.hpp"

namespace lbm {

namespace {
constexpr double rho0 = 1.0;

// Device constants for D2Q9
__constant__ double d_weights[9];
__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ int d_opposite[9];

// Initialize device constants
void init_device_constants() {
    static bool initialized = false;
    if (!initialized) {
        cudaMemcpyToSymbol(d_weights, D2Q9::weights.data(), 9 * sizeof(double));
        cudaMemcpyToSymbol(d_cx, D2Q9::cx.data(), 9 * sizeof(int));
        cudaMemcpyToSymbol(d_cy, D2Q9::cy.data(), 9 * sizeof(int));
        cudaMemcpyToSymbol(d_opposite, D2Q9::opposite.data(), 9 * sizeof(int));
        initialized = true;
    }
}

__device__ std::size_t lattice_index(std::size_t x, std::size_t y, std::size_t q, std::size_t nx) {
    return (y * nx + x) * 9 + q;
}

__device__ std::size_t cell_index(std::size_t x, std::size_t y, std::size_t nx) {
    return y * nx + x;
}

__global__ void initialize_lattice_kernel(double* f_curr, std::size_t nx, std::size_t ny) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) {
        return;
    }
    
    const std::size_t base = cell_index(x, y, nx) * 9;
    for (int q = 0; q < 9; ++q) {
        f_curr[base + q] = d_weights[q] * rho0;
    }
}

// Standard collide-and-stream kernel
__global__ void collide_and_stream_kernel(
    const double* f_curr,
    double* f_next,
    const bool* obstacle_mask,
    double omega,
    std::size_t nx,
    std::size_t ny) {
    
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) {
        return;
    }
    
    const std::size_t cell = cell_index(x, y, nx);
    
    // Skip obstacle cells (they're handled by boundary conditions)
    if (obstacle_mask && obstacle_mask[cell]) {
        return;
    }
    
    // Compute macroscopic quantities
    double rho = 0.0;
    double ux = 0.0;
    double uy = 0.0;
    
    for (int q = 0; q < 9; ++q) {
        const double fval = f_curr[lattice_index(x, y, q, nx)];
        rho += fval;
        ux += fval * static_cast<double>(d_cx[q]);
        uy += fval * static_cast<double>(d_cy[q]);
    }
    
    ux /= rho;
    uy /= rho;
    
    const double u_sq = ux * ux + uy * uy;
    
    // Collision and streaming
    for (int q = 0; q < 9; ++q) {
        const double e_dot_u = d_cx[q] * ux + d_cy[q] * uy;
        const double feq = d_weights[q] * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
        const double f_post = f_curr[lattice_index(x, y, q, nx)] + omega * (feq - f_curr[lattice_index(x, y, q, nx)]);
        
        const int xn = static_cast<int>(x) + d_cx[q];
        const int yn = static_cast<int>(y) + d_cy[q];
        
        if (xn >= 0 && yn >= 0 && xn < static_cast<int>(nx) && yn < static_cast<int>(ny)) {
            const std::size_t target_cell = cell_index(static_cast<std::size_t>(xn), static_cast<std::size_t>(yn), nx);
            if (obstacle_mask && obstacle_mask[target_cell]) {
                // Bounce-back at obstacle
                const int qo = d_opposite[q];
                f_next[lattice_index(x, y, qo, nx)] = f_post;
            } else {
                f_next[lattice_index(static_cast<std::size_t>(xn), static_cast<std::size_t>(yn), q, nx)] = f_post;
            }
        } else {
            // Bounce-back at domain boundary
            const int qo = d_opposite[q];
            f_next[lattice_index(x, y, qo, nx)] = f_post;
        }
    }
}

// Optimized collide-and-stream kernel with shared memory tiling
__global__ void collide_and_stream_tiled_kernel(
    const double* f_curr,
    double* f_next,
    const bool* obstacle_mask,
    double omega,
    std::size_t nx,
    std::size_t ny) {
    
    // Shared memory for tile of distribution functions
    // Tile size: (TILE_X + 2) x (TILE_Y + 2) to include halo
    constexpr int TILE_X = 16;
    constexpr int TILE_Y = 16;
    constexpr int HALO = 1;
    
    __shared__ double tile_f[TILE_Y + 2 * HALO][TILE_X + 2 * HALO][9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Global position
    const int gx = bx * TILE_X + tx - HALO;
    const int gy = by * TILE_Y + ty - HALO;
    
    // Load tile into shared memory (with halo)
    if (gx >= 0 && gx < static_cast<int>(nx) && gy >= 0 && gy < static_cast<int>(ny)) {
        const std::size_t cell = cell_index(static_cast<std::size_t>(gx), static_cast<std::size_t>(gy), nx);
        for (int q = 0; q < 9; ++q) {
            tile_f[ty][tx][q] = f_curr[lattice_index(static_cast<std::size_t>(gx), static_cast<std::size_t>(gy), q, nx)];
        }
    } else {
        for (int q = 0; q < 9; ++q) {
            tile_f[ty][tx][q] = 0.0;
        }
    }
    
    __syncthreads();
    
    // Process interior of tile (avoid halo)
    if (tx >= HALO && tx < TILE_X + HALO && ty >= HALO && ty < TILE_Y + HALO) {
        const int x = bx * TILE_X + tx - HALO;
        const int y = by * TILE_Y + ty - HALO;
        
        if (x >= 0 && y >= 0 && x < static_cast<int>(nx) && y < static_cast<int>(ny)) {
            const std::size_t cell = cell_index(static_cast<std::size_t>(x), static_cast<std::size_t>(y), nx);
            
            if (obstacle_mask && obstacle_mask[cell]) {
                return;
            }
            
            // Compute macroscopic quantities from shared memory
            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            
            for (int q = 0; q < 9; ++q) {
                const double fval = tile_f[ty][tx][q];
                rho += fval;
                ux += fval * static_cast<double>(d_cx[q]);
                uy += fval * static_cast<double>(d_cy[q]);
            }
            
            ux /= rho;
            uy /= rho;
            const double u_sq = ux * ux + uy * uy;
            
            // Collision
            for (int q = 0; q < 9; ++q) {
                const double e_dot_u = d_cx[q] * ux + d_cy[q] * uy;
                const double feq = d_weights[q] * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
                const double f_post = tile_f[ty][tx][q] + omega * (feq - tile_f[ty][tx][q]);
                
                // Streaming (write to global memory)
                const int xn = x + d_cx[q];
                const int yn = y + d_cy[q];
                
                if (xn >= 0 && yn >= 0 && xn < static_cast<int>(nx) && yn < static_cast<int>(ny)) {
                    const std::size_t target_cell = cell_index(static_cast<std::size_t>(xn), static_cast<std::size_t>(yn), nx);
                    if (obstacle_mask && obstacle_mask[target_cell]) {
                        const int qo = d_opposite[q];
                        f_next[lattice_index(static_cast<std::size_t>(x), static_cast<std::size_t>(y), qo, nx)] = f_post;
                    } else {
                        f_next[lattice_index(static_cast<std::size_t>(xn), static_cast<std::size_t>(yn), q, nx)] = f_post;
                    }
                } else {
                    const int qo = d_opposite[q];
                    f_next[lattice_index(static_cast<std::size_t>(x), static_cast<std::size_t>(y), qo, nx)] = f_post;
                }
            }
        }
    }
}

__global__ void apply_lid_velocity_kernel(
    double* f_curr,
    const bool* obstacle_mask,
    double u_lid,
    std::size_t nx,
    std::size_t ny) {
    
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t top = ny - 1;
    
    if (x >= nx) {
        return;
    }
    
    const std::size_t cell_idx = cell_index(x, top, nx);
    if (obstacle_mask && obstacle_mask[cell_idx]) {
        return;
    }
    
    const std::size_t base = cell_idx * 9;
    
    // Compute density
    double rho = 0.0;
    for (int q = 0; q < 9; ++q) {
        rho += f_curr[base + q];
    }
    
    const double u_sq = u_lid * u_lid;
    
    // Apply lid velocity boundary condition
    for (int q = 0; q < 9; ++q) {
        if (d_cy[q] == -1) {  // Directions pointing downward from the lid
            const double e_dot_u = d_cx[q] * u_lid;
            const double feq = d_weights[q] * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
            f_curr[base + q] = feq;
        }
    }
}

__global__ void apply_inflow_outflow_kernel(
    double* f_curr,
    const bool* obstacle_mask,
    double u_inflow,
    std::size_t nx,
    std::size_t ny) {
    
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t cell_count = nx * ny;
    
    if (idx >= cell_count) {
        return;
    }
    
    const std::size_t x = idx % nx;
    const std::size_t y = idx / nx;
    const std::size_t cell = cell_index(x, y, nx);
    
    if (obstacle_mask && obstacle_mask[cell]) {
        return;
    }
    
    const std::size_t base = cell * 9;
    const double rho_inflow = 1.0;
    const double u_sq = u_inflow * u_inflow;
    
    // Inflow boundary (left wall, x=0)
    if (x == 0) {
        for (int q = 0; q < 9; ++q) {
            if (d_cx[q] == 1) {  // Directions pointing right (inflow direction)
                const double e_dot_u = d_cx[q] * u_inflow;
                const double feq = d_weights[q] * rho_inflow * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
                f_curr[base + q] = feq;
            }
        }
    }
    
    // Outflow boundary (right wall, x=nx-1) - zero gradient
    if (x == nx - 1) {
        const std::size_t cell_left = cell_index(x - 1, y, nx);
        for (int q = 0; q < 9; ++q) {
            if (d_cx[q] == -1) {  // Directions pointing left (outflow direction)
                f_curr[base + q] = f_curr[cell_left * 9 + q];
            }
        }
    }
    
    // No-slip walls (top and bottom)
    if (y == 0) {
        // Bottom wall
        for (int q = 0; q < 9; ++q) {
            if (d_cy[q] == -1) {  // Directions pointing down
                const int qo = d_opposite[q];
                f_curr[base + q] = f_curr[base + qo];
            }
        }
    }
    
    if (y == ny - 1) {
        // Top wall
        for (int q = 0; q < 9; ++q) {
            if (d_cy[q] == 1) {  // Directions pointing up
                const int qo = d_opposite[q];
                f_curr[base + q] = f_curr[base + qo];
            }
        }
    }
}

__global__ void compute_macro_fields_kernel(
    const double* f_curr,
    double* density,
    double* ux,
    double* uy,
    std::size_t nx,
    std::size_t ny) {
    
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) {
        return;
    }
    
    const std::size_t cell = cell_index(x, y, nx);
    
    double rho = 0.0;
    double ux_val = 0.0;
    double uy_val = 0.0;
    
    for (int q = 0; q < 9; ++q) {
        const double fval = f_curr[lattice_index(x, y, q, nx)];
        rho += fval;
        ux_val += fval * static_cast<double>(d_cx[q]);
        uy_val += fval * static_cast<double>(d_cy[q]);
    }
    
    ux_val /= rho;
    uy_val /= rho;
    
    density[cell] = rho;
    ux[cell] = ux_val;
    uy[cell] = uy_val;
}

// Residual computation is done on host after copying velocity fields

}  // namespace

void CudaLbBackend::initialize(const SimulationConfig& config) {
    config_ = config;
    timestep_ = 0;
    residual_ = std::numeric_limits<double>::infinity();
    
    init_device_constants();
    release_device_buffers();
    
    // Create CUDA streams
    cudaStreamCreate(&compute_stream_);
    cudaStreamCreate(&transfer_stream_);
    
    allocate_device_buffers(config);
    initialize_lattice();
    build_obstacle_mask();
}

void CudaLbBackend::step() {
    if (!f_curr_ || is_converged()) {
        return;
    }
    
    collide_and_stream();
    apply_boundary_conditions();
    compute_macro_fields();
    
    ++timestep_;
}

bool CudaLbBackend::is_converged() const {
    if (timestep_ >= config_.max_timesteps) {
        return true;
    }
    return residual_ <= config_.residual_tolerance;
}

std::size_t CudaLbBackend::current_timestep() const {
    return timestep_;
}

DiagnosticSnapshot CudaLbBackend::fetch_diagnostics() const {
    DiagnosticSnapshot snapshot{};
    snapshot.timestep = timestep_;
    snapshot.residual_l2 = residual_;
    
    // Compute simple drag/lift proxies as domain-averaged velocities
    if (ux_ && uy_) {
        const std::size_t cell_count = config_.nx * config_.ny;
        std::vector<double> h_ux(cell_count);
        std::vector<double> h_uy(cell_count);
        
        cudaMemcpy(h_ux.data(), ux_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), uy_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
        
        const double mean_ux = std::accumulate(h_ux.begin(), h_ux.end(), 0.0) / static_cast<double>(cell_count);
        const double mean_uy = std::accumulate(h_uy.begin(), h_uy.end(), 0.0) / static_cast<double>(cell_count);
        
        snapshot.drag_coefficient = mean_ux;
        snapshot.lift_coefficient = mean_uy;
    }
    
    return snapshot;
}

void CudaLbBackend::allocate_device_buffers(const SimulationConfig& config) {
    const std::size_t cell_count = config.nx * config.ny;
    const std::size_t lattice_count = cell_count * 9;
    const std::size_t lattice_bytes = lattice_count * sizeof(double);
    const std::size_t cell_bytes = cell_count * sizeof(double);
    const std::size_t mask_bytes = cell_count * sizeof(bool);
    
    if (cudaMalloc(&f_curr_, lattice_bytes) != cudaSuccess ||
        cudaMalloc(&f_next_, lattice_bytes) != cudaSuccess ||
        cudaMalloc(&density_, cell_bytes) != cudaSuccess ||
        cudaMalloc(&ux_, cell_bytes) != cudaSuccess ||
        cudaMalloc(&uy_, cell_bytes) != cudaSuccess ||
        cudaMalloc(&obstacle_mask_, mask_bytes) != cudaSuccess) {
        release_device_buffers();
        throw std::runtime_error("Failed to allocate device buffers");
    }
    
    // Initialize f_next to zero
    cudaMemset(f_next_, 0, lattice_bytes);
    cudaMemset(density_, 0, cell_bytes);
    cudaMemset(ux_, 0, cell_bytes);
    cudaMemset(uy_, 0, cell_bytes);
    cudaMemset(obstacle_mask_, 0, mask_bytes);
}

void CudaLbBackend::release_device_buffers() {
    if (f_curr_) {
        cudaFree(f_curr_);
        f_curr_ = nullptr;
    }
    if (f_next_) {
        cudaFree(f_next_);
        f_next_ = nullptr;
    }
    if (density_) {
        cudaFree(density_);
        density_ = nullptr;
    }
    if (ux_) {
        cudaFree(ux_);
        ux_ = nullptr;
    }
    if (uy_) {
        cudaFree(uy_);
        uy_ = nullptr;
    }
    if (obstacle_mask_) {
        cudaFree(obstacle_mask_);
        obstacle_mask_ = nullptr;
    }
}

void CudaLbBackend::initialize_lattice() {
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (config_.nx + block_dim.x - 1) / block_dim.x,
        (config_.ny + block_dim.y - 1) / block_dim.y);
    
    initialize_lattice_kernel<<<grid_dim, block_dim>>>(f_curr_, config_.nx, config_.ny);
    cudaDeviceSynchronize();
}

void CudaLbBackend::build_obstacle_mask() {
    const std::size_t cell_count = config_.nx * config_.ny;
    std::vector<bool> host_mask(cell_count, false);
    
    for (const auto& obstacle : config_.obstacles) {
        if (obstacle.type == "cylinder") {
            // Parameters: [cx, cy, radius] in lattice units
            if (obstacle.parameters.size() >= 3) {
                const double cx = obstacle.parameters[0];
                const double cy = obstacle.parameters[1];
                const double radius = obstacle.parameters[2];
                
                for (std::size_t y = 0; y < config_.ny; ++y) {
                    for (std::size_t x = 0; x < config_.nx; ++x) {
                        const double dx = static_cast<double>(x) - cx;
                        const double dy = static_cast<double>(y) - cy;
                        if ((dx * dx + dy * dy) <= (radius * radius)) {
                            host_mask[y * config_.nx + x] = true;
                        }
                    }
                }
            }
        }
    }
    
    // Convert vector<bool> to array for CUDA transfer (vector<bool> is specialized)
    std::vector<char> host_mask_array(cell_count);
    for (std::size_t i = 0; i < cell_count; ++i) {
        host_mask_array[i] = host_mask[i] ? 1 : 0;
    }
    
    cudaMemcpy(obstacle_mask_, host_mask_array.data(), cell_count * sizeof(bool), cudaMemcpyHostToDevice);
}

void CudaLbBackend::collide_and_stream() {
    if (use_tiled_kernel_) {
        collide_and_stream_tiled();
    } else {
        collide_and_stream_standard();
    }
}

void CudaLbBackend::collide_and_stream_standard() {
    const double omega = 1.0 / config_.relaxation_time;
    
    // Clear f_next
    const std::size_t lattice_count = config_.nx * config_.ny * 9;
    cudaMemset(f_next_, 0, lattice_count * sizeof(double));
    
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (config_.nx + block_dim.x - 1) / block_dim.x,
        (config_.ny + block_dim.y - 1) / block_dim.y);
    
    collide_and_stream_kernel<<<grid_dim, block_dim>>>(
        f_curr_, f_next_, obstacle_mask_, omega, config_.nx, config_.ny);
    cudaDeviceSynchronize();
    
    // Swap buffers
    std::swap(f_curr_, f_next_);
}

void CudaLbBackend::collide_and_stream_tiled() {
    const double omega = 1.0 / config_.relaxation_time;
    
    // Clear f_next (async on compute stream)
    const std::size_t lattice_count = config_.nx * config_.ny * 9;
    cudaMemsetAsync(f_next_, 0, lattice_count * sizeof(double), compute_stream_);
    
    // Tiled kernel: 16x16 threads per block, processing 16x16 cells
    constexpr int TILE_SIZE = 16;
    const dim3 block_dim(TILE_SIZE + 2, TILE_SIZE + 2);  // +2 for halo
    const dim3 grid_dim(
        (config_.nx + TILE_SIZE - 1) / TILE_SIZE,
        (config_.ny + TILE_SIZE - 1) / TILE_SIZE);
    
    collide_and_stream_tiled_kernel<<<grid_dim, block_dim, 0, compute_stream_>>>(
        f_curr_, f_next_, obstacle_mask_, omega, config_.nx, config_.ny);
    
    // Swap buffers (synchronize stream first)
    cudaStreamSynchronize(compute_stream_);
    std::swap(f_curr_, f_next_);
}

void CudaLbBackend::apply_boundary_conditions() {
    // Apply lid velocity if configured (for cavity flow)
    if (config_.lid_velocity > 0.0 && config_.obstacles.empty()) {
        apply_lid_velocity();
    }
    
    // Apply inflow/outflow for cylinder flow
    if (!config_.obstacles.empty()) {
        apply_inflow_outflow();
    }
}

void CudaLbBackend::apply_lid_velocity() {
    const dim3 block_dim(256);
    const dim3 grid_dim((config_.nx + block_dim.x - 1) / block_dim.x);
    
    apply_lid_velocity_kernel<<<grid_dim, block_dim>>>(
        f_curr_, obstacle_mask_, config_.lid_velocity, config_.nx, config_.ny);
    cudaDeviceSynchronize();
}

void CudaLbBackend::apply_inflow_outflow() {
    const double u_inflow = 0.1;
    const std::size_t cell_count = config_.nx * config_.ny;
    const dim3 block_dim(256);
    const dim3 grid_dim((cell_count + block_dim.x - 1) / block_dim.x);
    
    apply_inflow_outflow_kernel<<<grid_dim, block_dim>>>(
        f_curr_, obstacle_mask_, u_inflow, config_.nx, config_.ny);
    cudaDeviceSynchronize();
}

void CudaLbBackend::compute_macro_fields() {
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (config_.nx + block_dim.x - 1) / block_dim.x,
        (config_.ny + block_dim.y - 1) / block_dim.y);
    
    compute_macro_fields_kernel<<<grid_dim, block_dim>>>(
        f_curr_, density_, ux_, uy_, config_.nx, config_.ny);
    cudaDeviceSynchronize();
    
    // Compute residual on host (simpler than atomic reduction)
    const std::size_t cell_count = config_.nx * config_.ny;
    std::vector<double> h_ux(cell_count);
    std::vector<double> h_uy(cell_count);
    
    cudaMemcpy(h_ux.data(), ux_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), uy_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
    
    residual_ = 0.0;
    for (std::size_t i = 0; i < cell_count; ++i) {
        const double vel_mag = std::sqrt(h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i]);
        residual_ = std::max(residual_, vel_mag);
    }
}

void CudaLbBackend::get_field_data(
    std::vector<double>& density,
    std::vector<double>& ux,
    std::vector<double>& uy) const {
    
    const std::size_t cell_count = config_.nx * config_.ny;
    
    if (!density_ || !ux_ || !uy_) {
        // Fields not allocated - return empty vectors
        density.clear();
        ux.clear();
        uy.clear();
        return;
    }
    
    density.resize(cell_count);
    ux.resize(cell_count);
    uy.resize(cell_count);
    
    // Ensure all previous kernels have completed
    cudaDeviceSynchronize();
    
    // Copy from device to host
    cudaError_t err1 = cudaMemcpy(density.data(), density_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaError_t err2 = cudaMemcpy(ux.data(), ux_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaError_t err3 = cudaMemcpy(uy.data(), uy_, cell_count * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Check for errors
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        // If copy failed, clear vectors to indicate failure
        density.clear();
        ux.clear();
        uy.clear();
    }
}

void CudaLbBackend::get_distributions(std::vector<double>& f_curr) const {
    const std::size_t lattice_count = config_.nx * config_.ny * 9;
    f_curr.resize(lattice_count);
    
    if (f_curr_) {
        cudaMemcpy(f_curr.data(), f_curr_, lattice_count * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

void CudaLbBackend::get_obstacle_mask(std::vector<bool>& obstacle_mask) const {
    const std::size_t cell_count = config_.nx * config_.ny;
    obstacle_mask.resize(cell_count);
    
    if (obstacle_mask_) {
        // Convert to array first (vector<bool> is specialized)
        std::vector<char> host_mask_array(cell_count);
        cudaMemcpy(host_mask_array.data(), obstacle_mask_, cell_count * sizeof(bool), cudaMemcpyDeviceToHost);
        
        // Convert back to vector<bool>
        for (std::size_t i = 0; i < cell_count; ++i) {
            obstacle_mask[i] = (host_mask_array[i] != 0);
        }
    }
}

CudaLbBackend::~CudaLbBackend() {
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
    }
    if (transfer_stream_) {
        cudaStreamDestroy(transfer_stream_);
    }
    release_device_buffers();
}

}  // namespace lbm
