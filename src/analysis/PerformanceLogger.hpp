#pragma once

#include <chrono>
#include <cstddef>
#include <fstream>
#include <string>

namespace lbm {

class PerformanceLogger {
  public:
    explicit PerformanceLogger(std::string output_path);

    void start_section(const std::string& label);
    void end_section();
    void write_snapshot(std::size_t timestep, double milliseconds);

  private:
    std::string output_path_;
    std::ofstream stream_;
    std::chrono::high_resolution_clock::time_point section_start_{};
};

}  // namespace lbm
