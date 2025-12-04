#include "VtkWriter.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lbm {

VtkWriter::VtkWriter(std::filesystem::path output_dir) : output_dir_(std::move(output_dir)) {
    std::filesystem::create_directories(output_dir_);
}

void VtkWriter::write_field(
    const std::string& filename,
    std::size_t nx,
    std::size_t ny,
    const double* density,
    const double* ux,
    const double* uy,
    std::size_t timestep) const {
    
    std::filesystem::path filepath = output_dir_ / filename;
    std::ofstream out(filepath);
    
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open VTK file: " + filepath.string());
    }
    
    // Write VTK header
    out << "# vtk DataFile Version 2.0\n";
    out << "LBM Flow Field - Timestep " << timestep << "\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nx << " " << ny << " 1\n";
    out << "ORIGIN 0.0 0.0 0.0\n";
    out << "SPACING 1.0 1.0 1.0\n";
    out << "POINT_DATA " << (nx * ny) << "\n";
    
    // Write density
    out << "SCALARS density double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const std::size_t idx = y * nx + x;
            out << std::scientific << std::setprecision(8) << density[idx] << "\n";
        }
    }
    
    // Write velocity
    out << "VECTORS velocity double\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const std::size_t idx = y * nx + x;
            out << std::scientific << std::setprecision(8)
                << ux[idx] << " " << uy[idx] << " 0.0\n";
        }
    }
    
    // Write velocity magnitude
    out << "SCALARS velocity_magnitude double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const std::size_t idx = y * nx + x;
            const double mag = std::sqrt(ux[idx] * ux[idx] + uy[idx] * uy[idx]);
            out << std::scientific << std::setprecision(8) << mag << "\n";
        }
    }
}

}  // namespace lbm

