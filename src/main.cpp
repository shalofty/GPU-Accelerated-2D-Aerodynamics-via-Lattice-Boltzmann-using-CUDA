#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "backend/cpu/CpuLbBackend.hpp"
#include "core/SimulationRunner.hpp"
#include "io/SimulationConfigBuilder.hpp"
#include "analysis/PerformanceLogger.hpp"
#include "analysis/VtkObserver.hpp"

#ifdef ENABLE_CUDA
#include "backend/cuda/CudaLbBackend.hpp"
#endif

namespace {
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --config <file>       Configuration file (YAML)\n"
              << "  --backend <cpu|cuda>  Backend to use (default: from config)\n"
              << "  --output-dir <dir>    Output directory for VTK files\n"
              << "  --help                Show this help message\n";
}

std::unique_ptr<lbm::SimulationBackend> create_backend(const std::string& backend_id) {
    if (backend_id == "cpu") {
        return std::make_unique<lbm::CpuLbBackend>();
    }
#ifdef ENABLE_CUDA
    else if (backend_id == "cuda") {
        return std::make_unique<lbm::CudaLbBackend>();
    }
#endif
    else {
        throw std::runtime_error("Unknown backend: " + backend_id);
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string config_path;
    std::string backend_override;
    std::string output_dir = "output";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            backend_override = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    try {
        // Load configuration
        lbm::SimulationConfigBuilder builder;
        if (!config_path.empty()) {
            builder.set_config_path(config_path);
        } else {
            std::cerr << "Error: --config required\n";
            print_usage(argv[0]);
            return 1;
        }
        
        auto config = builder.build();
        
        // Override backend if specified
        if (!backend_override.empty()) {
            config.backend_id = backend_override;
        }
        
        std::cout << "=== LBM Simulation ===\n";
        std::cout << "Grid: " << config.nx << "x" << config.ny << "\n";
        std::cout << "Backend: " << config.backend_id << "\n";
        std::cout << "Max timesteps: " << config.max_timesteps << "\n";
        std::cout << "Relaxation time: " << config.relaxation_time << "\n";
        std::cout << "Lid velocity: " << config.lid_velocity << "\n";
        std::cout << "Obstacles: " << config.obstacles.size() << "\n";
        for (const auto& obs : config.obstacles) {
            std::cout << "  - " << obs.id << " (" << obs.type << "): ";
            for (double p : obs.parameters) {
                std::cout << p << " ";
            }
            std::cout << "\n";
        }
        std::cout << "Output directory: " << output_dir << "\n";
        std::cout << "\n";
        
        // Create backend
        auto backend = create_backend(config.backend_id);
        
        // Create runner
        lbm::SimulationRunner runner(std::move(backend));
        
        // Add observers
        auto perf_logger = std::make_shared<lbm::PerformanceLogger>(output_dir + "/performance.csv");
        runner.add_observer(perf_logger);
        
        auto vtk_observer = std::make_shared<lbm::VtkObserver>(
            output_dir, config.output_interval, runner.backend());
        runner.add_observer(vtk_observer);
        
        // Run simulation
        std::cout << "Starting simulation...\n";
        const auto start_time = std::chrono::high_resolution_clock::now();
        
        perf_logger->start_section("simulation");
        runner.run(config);
        perf_logger->end_section();
        
        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration<double>(end_time - start_time).count();
        
        // Final diagnostics
        auto final_snapshot = runner.backend()->fetch_diagnostics();
        
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Final timestep: " << final_snapshot.timestep << "\n";
        std::cout << "Final residual: " << final_snapshot.residual_l2 << "\n";
        std::cout << "Drag coefficient: " << final_snapshot.drag_coefficient << "\n";
        std::cout << "Lift coefficient: " << final_snapshot.lift_coefficient << "\n";
        std::cout << "Total time: " << duration << " seconds\n";
        std::cout << "Time per timestep: " << (duration / final_snapshot.timestep) * 1000.0 << " ms\n";
        std::cout << "\n";
        std::cout << "Results saved to: " << output_dir << "/\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

