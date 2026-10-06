#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace facial_fire {

// Own both buffers. Each update reads current_, writes next_, then swaps.
class Simulation {
public:
    Simulation(int rows, int cols, const std::string& mode = "perimeter",
               std::uint64_t seed = 1);
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const std::vector<float>& grid() const { return current_; }
    void set_grid(const float* values);
    void reset();
    void ignite(float u, float v, float radius, float intensity);
    void step(float dt, float spread_speed, float cooling, int steps);

private:
    void update_serial(float dt, float spread_speed, float cooling);
    int rows_;
    int cols_;
    std::vector<float> current_;
    std::vector<float> next_;
    bool perimeter_;
    std::uint64_t seed_;
    std::uint64_t tick_ = 0;
};

}  // namespace facial_fire
