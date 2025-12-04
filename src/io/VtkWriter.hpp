#pragma once

#include <filesystem>
#include <string>

namespace lbm {

class VtkWriter {
  public:
    explicit VtkWriter(std::filesystem::path output_dir);
    
    void write_field(
        const std::string& filename,
        std::size_t nx,
        std::size_t ny,
        const double* density,
        const double* ux,
        const double* uy,
        std::size_t timestep) const;

  private:
    std::filesystem::path output_dir_;
};

}  // namespace lbm

