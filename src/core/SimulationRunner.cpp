#include "SimulationRunner.hpp"

#include <utility>

#include "../analysis/DiagnosticObserver.hpp"

namespace lbm {

SimulationRunner::SimulationRunner(SimulationBackendPtr backend)
    : backend_(std::move(backend)) {}

void SimulationRunner::add_observer(std::shared_ptr<DiagnosticObserver> observer) {
    observers_.push_back(std::move(observer));
}

void SimulationRunner::run(const SimulationConfig& config) {
    if (!backend_) {
        return;
    }

    backend_->initialize(config);

    while (!backend_->is_converged()) {
        backend_->step();
        notify_observers(backend_->fetch_diagnostics());
    }
}

void SimulationRunner::notify_observers(const DiagnosticSnapshot& snapshot) const {
    for (const auto& observer : observers_) {
        if (observer) {
            observer->on_step(snapshot);
        }
    }
}

}  // namespace lbm
