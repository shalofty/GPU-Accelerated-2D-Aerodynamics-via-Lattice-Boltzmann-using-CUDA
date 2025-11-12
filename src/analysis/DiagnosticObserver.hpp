#pragma once

#include <memory>

#include "../core/SimulationBackend.hpp"

namespace lbm {

class DiagnosticObserver {
  public:
    virtual ~DiagnosticObserver() = default;
    virtual void on_step(const DiagnosticSnapshot& snapshot) = 0;
};

using DiagnosticObserverPtr = std::shared_ptr<DiagnosticObserver>;

}  // namespace lbm
