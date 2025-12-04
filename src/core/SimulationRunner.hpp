#pragma once

#include <memory>
#include <vector>

#include "SimulationBackend.hpp"

namespace lbm {

class DiagnosticObserver;

class SimulationRunner {
  public:
    explicit SimulationRunner(SimulationBackendPtr backend);

    void add_observer(std::shared_ptr<DiagnosticObserver> observer);
    void run(const SimulationConfig& config);

    SimulationBackend* backend() { return backend_.get(); }

  private:
    void notify_observers(const DiagnosticSnapshot& snapshot) const;

    SimulationBackendPtr backend_;
    std::vector<std::shared_ptr<DiagnosticObserver>> observers_;
    
    friend class VtkObserver;  // Allow access to backend for field data
};

}  // namespace lbm
