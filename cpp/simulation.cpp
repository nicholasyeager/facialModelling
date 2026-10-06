#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace facial_fire {
namespace {
void require_range(float value, float low, float high, const char* name) {
    if (!std::isfinite(value) || value < low || value > high) {
        throw std::invalid_argument(name);
    }
}
}  // namespace

Simulation::Simulation(int rows, int cols) : rows_(rows), cols_(cols) {
    if (rows < 2 || cols < 2 || rows > 2048 || cols > 2048) {
        throw std::invalid_argument("Grid dimensions must be between 2 and 2048");
    }
    const auto count = static_cast<std::size_t>(rows) * cols;
    current_.assign(count, 0.0f);
    next_.assign(count, 0.0f);
}

void Simulation::set_grid(const float* values) {
    // Validate before mutation: invalid input must leave state intact.
    for (std::size_t i = 0; i < current_.size(); ++i) {
        require_range(values[i], 0.0f, 1.0f, "Grid intensities must be finite and in [0, 1]");
    }
    std::copy(values, values + current_.size(), current_.begin());
    std::fill(next_.begin(), next_.end(), 0.0f);
}

void Simulation::reset() {
    std::fill(current_.begin(), current_.end(), 0.0f);
    std::fill(next_.begin(), next_.end(), 0.0f);
}

void Simulation::ignite(float u, float v, float radius, float intensity) {
    require_range(u, 0.0f, 1.0f, "Ignition u must be in [0, 1]");
    require_range(v, 0.0f, 1.0f, "Ignition v must be in [0, 1]");
    require_range(radius, 0.0f, 1.0f, "Ignition radius must be in [0, 1]");
    require_range(intensity, 0.0f, 1.0f, "Ignition intensity must be in [0, 1]");
    for (int row = 0; row < rows_; ++row) {
        const float dy = static_cast<float>(row) / (rows_ - 1) - v;
        for (int col = 0; col < cols_; ++col) {
            const float dx = static_cast<float>(col) / (cols_ - 1) - u;
            if (dx * dx + dy * dy <= radius * radius) {
                auto& cell = current_[static_cast<std::size_t>(row) * cols_ + col];
                cell = std::max(cell, intensity);
            }
        }
    }
    // Always ignite at least the closest cell, including radius == 0.
    const int col = static_cast<int>(std::lround(u * (cols_ - 1)));
    const int row = static_cast<int>(std::lround(v * (rows_ - 1)));
    auto& closest = current_[static_cast<std::size_t>(row) * cols_ + col];
    closest = std::max(closest, intensity);
}

void Simulation::step(float dt, float spread_speed, float cooling, int steps) {
    require_range(dt, 0.0f, 1.0f, "Timestep must be finite and in [0, 1]");
    require_range(spread_speed, 0.0f, 60.0f, "Spread speed must be in [0, 60]");
    require_range(cooling, 0.0f, 60.0f, "Cooling must be in [0, 60]");
    if (steps < 0) {
        throw std::invalid_argument("Step count cannot be negative");
    }
    for (int iteration = 0; iteration < steps; ++iteration) {
        update_serial(dt, spread_speed, cooling);
    }
}

void Simulation::update_serial(float dt, float spread_speed, float cooling) {
    for (int row = 0; row < rows_; ++row) {
        for (int col = 0; col < cols_; ++col) {
            const auto i = static_cast<std::size_t>(row) * cols_ + col;
            float neighbor = 0.0f;  // Missing neighbors contribute zero; no wraparound.
            if (row > 0) neighbor = std::max(neighbor, current_[i - cols_]);
            if (row + 1 < rows_) neighbor = std::max(neighbor, current_[i + cols_]);
            if (col > 0) neighbor = std::max(neighbor, current_[i - 1]);
            if (col + 1 < cols_) neighbor = std::max(neighbor, current_[i + 1]);
            const float value = current_[i];
            const float growth = spread_speed * neighbor * (1.0f - value);
            next_[i] = std::clamp(value + dt * (growth - cooling * value), 0.0f, 1.0f);
        }
    }
    current_.swap(next_);
}

}  // namespace facial_fire
