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
    bool in_obstacles_section = false;
    ObstacleDefinition current_obstacle;
    bool in_obstacle_entry = false;
    
    while (std::getline(file, line)) {
        std::string original_line = line;
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check if we're entering the obstacles section
        if (line.find("obstacles:") == 0 || line.find("obstacles :") == 0) {
            in_obstacles_section = true;
            continue;
        }
        
        // If we're in obstacles section, parse obstacle entries
        if (in_obstacles_section) {
            // Check if this is the start of a new obstacle entry (starts with "-")
            if (line.find("-") == 0) {
                // Save previous obstacle if we have one
                if (in_obstacle_entry && !current_obstacle.id.empty()) {
                    config.obstacles.push_back(current_obstacle);
                }
                // Start new obstacle
                current_obstacle = ObstacleDefinition{};
                in_obstacle_entry = true;
                // Check if id is on the same line: "- id: ..."
                if (line.find("id:") != std::string::npos) {
                    std::string value = get_value(line.substr(line.find("id:")));
                    value = trim(value);
                    if (!value.empty() && value[0] == '"' && value.back() == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    current_obstacle.id = value;
                }
                continue;
            }
            
            // Parse obstacle fields (indented)
            if (in_obstacle_entry) {
                if (line.find("id:") == 0 || (line.size() > 4 && line.substr(0, 4) == "  id:")) {
                    std::string value = get_value(line);
                    value = trim(value);
                    if (!value.empty() && value[0] == '"' && value.back() == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    current_obstacle.id = value;
                } else if (line.find("type:") == 0 || (line.size() > 6 && line.substr(0, 6) == "  type:")) {
                    std::string value = get_value(line);
                    value = trim(value);
                    if (!value.empty() && value[0] == '"' && value.back() == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    current_obstacle.type = value;
                } else if (line.find("parameters:") != std::string::npos || 
                          (line.size() > 12 && line.substr(0, 12) == "  parameters:")) {
                    // Parse array: [100.0, 100.0, 25.0]
                    std::string value = get_value(line);
                    value = trim(value);
                    // Remove brackets
                    if (value[0] == '[') value = value.substr(1);
                    if (value.back() == ']') value = value.substr(0, value.length() - 1);
                    
                    // Split by comma and parse doubles
                    std::istringstream iss(value);
                    std::string token;
                    current_obstacle.parameters.clear();
                    while (std::getline(iss, token, ',')) {
                        token = trim(token);
                        if (!token.empty()) {
                            double param;
                            if (parse_double(token, param)) {
                                current_obstacle.parameters.push_back(param);
                            }
                        }
                    }
                } else if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
                    // We've left the obstacles section
                    in_obstacles_section = false;
                    if (in_obstacle_entry && !current_obstacle.id.empty()) {
                        config.obstacles.push_back(current_obstacle);
                        in_obstacle_entry = false;
                    }
                    // Fall through to parse this line as a regular key-value pair
                } else {
                    continue;
                }
                continue;
            }
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
    
    // Save last obstacle if we were still parsing one
    if (in_obstacle_entry && !current_obstacle.id.empty()) {
        config.obstacles.push_back(current_obstacle);
    }
    
    return config;
}

}  // namespace lbm
