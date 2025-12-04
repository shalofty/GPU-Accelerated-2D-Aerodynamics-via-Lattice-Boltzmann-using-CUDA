#include "CudaProfiler.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace lbm {

CudaProfiler::CudaProfiler() {
#ifdef ENABLE_CUDA
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
#endif
}

CudaProfiler::~CudaProfiler() {
#ifdef ENABLE_CUDA
    if (start_event_) {
        cudaEventDestroy(start_event_);
    }
    if (stop_event_) {
        cudaEventDestroy(stop_event_);
    }
#endif
}

void CudaProfiler::start_timer(const std::string& label) {
    (void)label;  // Unused when CUDA disabled
#ifdef ENABLE_CUDA
    cudaEventRecord(start_event_);
#endif
}

void CudaProfiler::stop_timer(const std::string& label) {
#ifdef ENABLE_CUDA
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    
    // Find or create timing entry
    auto it = std::find_if(timings_.begin(), timings_.end(),
        [&label](const KernelTiming& t) { return t.name == label; });
    
    if (it != timings_.end()) {
        it->elapsed_ms += static_cast<double>(elapsed_ms);
        it->call_count++;
    } else {
        KernelTiming timing;
        timing.name = label;
        timing.elapsed_ms = static_cast<double>(elapsed_ms);
        timing.call_count = 1;
        timings_.push_back(timing);
    }
#endif
}

void CudaProfiler::record_kernel_start() {
#ifdef ENABLE_CUDA
    cudaEventRecord(start_event_);
#else
    // No-op when CUDA disabled
#endif
}

void CudaProfiler::record_kernel_end(const std::string& kernel_name) {
#ifdef ENABLE_CUDA
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    
    // Find or create timing entry
    auto it = std::find_if(timings_.begin(), timings_.end(),
        [&kernel_name](const KernelTiming& t) { return t.name == kernel_name; });
    
    if (it != timings_.end()) {
        it->elapsed_ms += static_cast<double>(elapsed_ms);
        it->call_count++;
    } else {
        KernelTiming timing;
        timing.name = kernel_name;
        timing.elapsed_ms = static_cast<double>(elapsed_ms);
        timing.call_count = 1;
        timings_.push_back(timing);
    }
#endif
}

std::vector<KernelTiming> CudaProfiler::get_timings() const {
    return timings_;
}

void CudaProfiler::reset() {
    timings_.clear();
}

void CudaProfiler::write_report(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        return;
    }
    
    out << "CUDA Kernel Performance Report\n";
    out << "==============================\n\n";
    out << std::left << std::setw(30) << "Kernel Name"
        << std::setw(15) << "Total (ms)"
        << std::setw(15) << "Calls"
        << std::setw(15) << "Avg (ms)" << "\n";
    out << std::string(75, '-') << "\n";
    
    for (const auto& timing : timings_) {
        const double avg_ms = timing.call_count > 0 ? timing.elapsed_ms / timing.call_count : 0.0;
        out << std::left << std::setw(30) << timing.name
            << std::fixed << std::setprecision(3)
            << std::setw(15) << timing.elapsed_ms
            << std::setw(15) << timing.call_count
            << std::setw(15) << avg_ms << "\n";
    }
}

}  // namespace lbm

