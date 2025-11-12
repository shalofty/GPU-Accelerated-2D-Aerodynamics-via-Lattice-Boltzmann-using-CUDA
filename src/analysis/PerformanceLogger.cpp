#include "PerformanceLogger.hpp"

#include <iomanip>
#include <stdexcept>

namespace lbm {

namespace {
constexpr const char* kDefaultHeader = "label,start_ms,end_ms,duration_ms";
}

PerformanceLogger::PerformanceLogger(std::string output_path)
    : output_path_(std::move(output_path)), stream_(output_path_, std::ios::out | std::ios::trunc) {
    if (!stream_.is_open()) {
        throw std::runtime_error("Failed to open performance log: " + output_path_);
    }
    stream_ << kDefaultHeader << '\n';
}

void PerformanceLogger::start_section(const std::string& label) {
    stream_ << label << ',';
    section_start_ = std::chrono::high_resolution_clock::now();
    stream_ << 0.0 << ',';
}

void PerformanceLogger::end_section() {
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration<double, std::milli>(end_time - section_start_).count();
    stream_ << duration << ',' << duration << '\n';
    stream_.flush();
}

void PerformanceLogger::write_snapshot(std::size_t timestep, double milliseconds) {
    stream_ << "snapshot-" << timestep << ",0.0," << milliseconds << ',' << milliseconds << '\n';
    stream_.flush();
}

}  // namespace lbm
