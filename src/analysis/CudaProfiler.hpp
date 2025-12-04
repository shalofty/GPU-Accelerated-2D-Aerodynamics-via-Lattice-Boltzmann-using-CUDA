#pragma once

#include <cstddef>
#include <string>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace lbm {

struct KernelTiming {
    std::string name;
    double elapsed_ms{0.0};
    std::size_t call_count{0};
};

class CudaProfiler {
  public:
    CudaProfiler();
    ~CudaProfiler();
    
    void start_timer(const std::string& label);
    void stop_timer(const std::string& label);
    
    void record_kernel_start();
    void record_kernel_end(const std::string& kernel_name);
    
    std::vector<KernelTiming> get_timings() const;
    void reset();
    void write_report(const std::string& filename) const;

  private:
#ifdef ENABLE_CUDA
    cudaEvent_t start_event_{nullptr};
    cudaEvent_t stop_event_{nullptr};
#endif
    std::vector<KernelTiming> timings_;
};

}  // namespace lbm

