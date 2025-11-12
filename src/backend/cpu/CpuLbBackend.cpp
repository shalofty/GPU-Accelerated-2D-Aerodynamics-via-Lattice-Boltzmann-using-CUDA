#include "CpuLbBackend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "../../core/D2Q9.hpp"

namespace lbm {
namespace {
constexpr double rho0 = 1.0;
}

void CpuLbBackend::initialize(const SimulationConfig& config) {
    config_ = config;
    timestep_ = 0;
    const std::size_t cell_count = config_.nx * config_.ny;
    const std::size_t lattice_count = cell_count * D2Q9::q;

    f_curr_.assign(lattice_count, 0.0);
    f_next_.assign(lattice_count, 0.0);
    density_.assign(cell_count, rho0);
    ux_.assign(cell_count, 0.0);
    uy_.assign(cell_count, 0.0);

    initialize_lattice();
    residual_ = std::numeric_limits<double>::infinity();
}

void CpuLbBackend::step() {
    if (is_converged()) {
        return;
    }

    collide_and_stream();
    apply_lid_velocity();
    compute_macro_fields();

    ++timestep_;
}

bool CpuLbBackend::is_converged() const {
    if (timestep_ >= config_.max_timesteps) {
        return true;
    }
    return residual_ <= config_.residual_tolerance;
}

std::size_t CpuLbBackend::current_timestep() const {
    return timestep_;
}

DiagnosticSnapshot CpuLbBackend::fetch_diagnostics() const {
    DiagnosticSnapshot snapshot{};
    snapshot.timestep = timestep_;
    snapshot.residual_l2 = residual_;

    // Compute simple drag/lift proxies as domain-averaged velocities.
    const double mean_ux = std::accumulate(ux_.begin(), ux_.end(), 0.0) / static_cast<double>(ux_.size());
    const double mean_uy = std::accumulate(uy_.begin(), uy_.end(), 0.0) / static_cast<double>(uy_.size());
    snapshot.drag_coefficient = mean_ux;
    snapshot.lift_coefficient = mean_uy;
    return snapshot;
}

std::size_t CpuLbBackend::lattice_index(std::size_t x, std::size_t y, std::size_t q) const {
    return (y * config_.nx + x) * D2Q9::q + q;
}

std::size_t CpuLbBackend::cell_index(std::size_t x, std::size_t y) const {
    return y * config_.nx + x;
}

void CpuLbBackend::initialize_lattice() {
    for (std::size_t y = 0; y < config_.ny; ++y) {
        for (std::size_t x = 0; x < config_.nx; ++x) {
            const std::size_t base = cell_index(x, y) * D2Q9::q;
            for (int q = 0; q < D2Q9::q; ++q) {
                f_curr_[base + q] = D2Q9::weights[q] * rho0;
            }
        }
    }
}

void CpuLbBackend::collide_and_stream() {
    const double tau = config_.relaxation_time;
    const double omega = 1.0 / tau;

    std::fill(f_next_.begin(), f_next_.end(), 0.0);

    for (std::size_t y = 0; y < config_.ny; ++y) {
        for (std::size_t x = 0; x < config_.nx; ++x) {
            const std::size_t cell = cell_index(x, y);

            // Macroscopic quantities from current distributions.
            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            for (int q = 0; q < D2Q9::q; ++q) {
                const double fval = f_curr_[lattice_index(x, y, q)];
                rho += fval;
                ux += fval * static_cast<double>(D2Q9::cx[q]);
                uy += fval * static_cast<double>(D2Q9::cy[q]);
            }
            ux /= rho;
            uy /= rho;

            const double u_sq = ux * ux + uy * uy;

            for (int q = 0; q < D2Q9::q; ++q) {
                const double e_dot_u = D2Q9::cx[q] * ux + D2Q9::cy[q] * uy;
                const double feq = D2Q9::weights[q] * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
                const double f_post = f_curr_[lattice_index(x, y, q)] + omega * (feq - f_curr_[lattice_index(x, y, q)]);

                const int xn = static_cast<int>(x) + D2Q9::cx[q];
                const int yn = static_cast<int>(y) + D2Q9::cy[q];

                if (xn >= 0 && yn >= 0 && xn < static_cast<int>(config_.nx) && yn < static_cast<int>(config_.ny)) {
                    f_next_[lattice_index(static_cast<std::size_t>(xn), static_cast<std::size_t>(yn), q)] = f_post;
                } else {
                    // Bounce-back at domain boundary.
                    const int qo = D2Q9::opposite[q];
                    f_next_[lattice_index(x, y, qo)] = f_post;
                }
            }
        }
    }

    f_curr_.swap(f_next_);
}

void CpuLbBackend::apply_lid_velocity() {
    const double u_lid = config_.lid_velocity;
    const std::size_t top = config_.ny - 1;

    const double u_sq = u_lid * u_lid;

    for (std::size_t x = 0; x < config_.nx; ++x) {
        const std::size_t cell = cell_index(x, top);

        double rho = 0.0;
        for (int q = 0; q < D2Q9::q; ++q) {
            rho += f_curr_[cell * D2Q9::q + q];
        }

        for (int q = 0; q < D2Q9::q; ++q) {
            if (D2Q9::cy[q] == -1) {  // Directions pointing downward from the lid.
                const double e_dot_u = D2Q9::cx[q] * u_lid;
                const double feq = D2Q9::weights[q] * rho * (1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u_sq);
                f_curr_[cell * D2Q9::q + q] = feq;
            }
        }
    }
}

void CpuLbBackend::compute_macro_fields() {
    residual_ = 0.0;
    for (std::size_t y = 0; y < config_.ny; ++y) {
        for (std::size_t x = 0; x < config_.nx; ++x) {
            const std::size_t cell = cell_index(x, y);

            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            for (int q = 0; q < D2Q9::q; ++q) {
                const double fval = f_curr_[lattice_index(x, y, q)];
                rho += fval;
                ux += fval * static_cast<double>(D2Q9::cx[q]);
                uy += fval * static_cast<double>(D2Q9::cy[q]);
            }

            ux /= rho;
            uy /= rho;

            density_[cell] = rho;
            ux_[cell] = ux;
            uy_[cell] = uy;

            residual_ = std::max(residual_, std::sqrt(ux * ux + uy * uy));
        }
    }
}

}  // namespace lbm
