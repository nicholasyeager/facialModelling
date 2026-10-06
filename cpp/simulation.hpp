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
    static bool openmp_available();
    bool parallel() const { return parallel_; }
    int threads() const { return threads_; }
    int last_threads() const { return last_threads_; }
    void set_execution(bool parallel, int threads);
    void set_grid(const float* values);
    void reset();
    void ignite(float u, float v, float radius, float intensity);
    void step(float dt, float spread_speed, float cooling, int steps);

private:
    void update_serial(float dt, float spread_speed, float cooling);
    void update_parallel(float dt, float spread_speed, float cooling);
    float update_cell(int row, int col, float dt, float spread_speed, float cooling) const;
    int rows_;
    int cols_;
    std::vector<float> current_;
    std::vector<float> next_;
    bool perimeter_;
    std::uint64_t seed_;
    std::uint64_t tick_ = 0;
    bool parallel_ = false;
    int threads_ = 4;
    int last_threads_ = 1;
};

}  // namespace facial_fire
