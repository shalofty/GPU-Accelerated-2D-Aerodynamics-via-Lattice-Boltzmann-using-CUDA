#include "SimulationConfigBuilder.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>

namespace lbm {

SimulationConfigBuilder& SimulationConfigBuilder::set_config_path(std::filesystem::path path) {
    config_path_ = std::move(path);
    return *this;
}

namespace {
// Simple YAML parser helpers
std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    auto end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

std::string get_value(const std::string& line) {
    auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) return "";
    return trim(line.substr(colon_pos + 1));
}

bool parse_double(const std::string& str, double& value) {
    try {
        value = std::stod(str);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_t(const std::string& str, std::size_t& value) {
    try {
        value = static_cast<std::size_t>(std::stoull(str));
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace

SimulationConfig SimulationConfigBuilder::build() const {
    SimulationConfig config{};
    
    // Default values
    config.nx = 256;
    config.ny = 256;
    config.relaxation_time = 0.6;
    config.max_timesteps = 1000;
    config.output_interval = 100;
    config.lid_velocity = 0.1;
    config.residual_tolerance = 1e-6;
    config.backend_id = "cuda";
    
    if (config_path_.empty()) {
        throw std::runtime_error("Configuration path not set");
    }
    
    if (!std::filesystem::exists(config_path_)) {
        throw std::runtime_error("Configuration file not found: " + config_path_.string());
    }
    
    std::ifstream file(config_path_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + config_path_.string());
    }
    
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key-value pairs
        if (line.find("nx:") == 0 || line.find("nx :") == 0) {
            std::size_t value;
            if (parse_size_t(get_value(line), value)) {
                config.nx = value;
            }
        } else if (line.find("ny:") == 0 || line.find("ny :") == 0) {
            std::size_t value;
            if (parse_size_t(get_value(line), value)) {
                config.ny = value;
            }
        } else if (line.find("relaxation_time:") == 0 || line.find("relaxation_time :") == 0) {
            double value;
            if (parse_double(get_value(line), value)) {
                config.relaxation_time = value;
            }
        } else if (line.find("max_timesteps:") == 0 || line.find("max_timesteps :") == 0) {
            std::size_t value;
            if (parse_size_t(get_value(line), value)) {
                config.max_timesteps = value;
            }
        } else if (line.find("output_interval:") == 0 || line.find("output_interval :") == 0) {
            std::size_t value;
            if (parse_size_t(get_value(line), value)) {
                config.output_interval = value;
            }
        } else if (line.find("lid_velocity:") == 0 || line.find("lid_velocity :") == 0) {
            double value;
            if (parse_double(get_value(line), value)) {
                config.lid_velocity = value;
            }
        } else if (line.find("residual_tolerance:") == 0 || line.find("residual_tolerance :") == 0) {
            double value;
            if (parse_double(get_value(line), value)) {
                config.residual_tolerance = value;
            }
        } else if (line.find("backend_id:") == 0 || line.find("backend_id :") == 0) {
            std::string value = trim(get_value(line));
            // Remove quotes if present
            if (!value.empty() && value[0] == '"' && value.back() == '"') {
                value = value.substr(1, value.length() - 2);
            }
            config.backend_id = value;
        }
    }
    
    return config;
}

}  // namespace lbm
