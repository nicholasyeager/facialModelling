#include "simulation.hpp"

#include <algorithm>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using facial_fire::Simulation;

PYBIND11_MODULE(_native, module) {
    module.doc() = "Equivalent serial/OpenMP seeded double-buffered propagation";
    module.attr("openmp_available") = Simulation::openmp_available();
    py::class_<Simulation>(module, "Simulation")
        .def(py::init<int, int, const std::string&, std::uint64_t>(),
             py::arg("rows"), py::arg("cols"), py::arg("mode") = "perimeter", py::arg("seed") = 1)
        .def_property_readonly("rows", &Simulation::rows)
        .def_property_readonly("cols", &Simulation::cols)
        .def_property_readonly("parallel", &Simulation::parallel)
        .def_property_readonly("threads", &Simulation::threads)
        .def_property_readonly("last_threads", &Simulation::last_threads)
        .def("set_execution", &Simulation::set_execution, py::arg("parallel"), py::arg("threads"))
        .def("reset", &Simulation::reset)
        .def("ignite", &Simulation::ignite, py::arg("u"), py::arg("v"),
             py::arg("radius") = 0.025f, py::arg("intensity") = 1.0f)
        .def("step", &Simulation::step, py::arg("dt"), py::arg("spread_speed"),
             py::arg("cooling"), py::arg("steps") = 1)
        .def("set_grid", [](Simulation& simulation,
                              py::array_t<float, py::array::c_style | py::array::forcecast> values) {
            if (values.ndim() != 2 || values.shape(0) != simulation.rows() ||
                values.shape(1) != simulation.cols()) {
                throw std::invalid_argument("Input grid shape must match simulation");
            }
            simulation.set_grid(values.data());
        }, py::arg("values"))
        .def("snapshot", [](const Simulation& simulation) {
            // Owned copy: Python cannot mutate buffers or retain a dangling view after swap.
            py::array_t<float> result({simulation.rows(), simulation.cols()});
            std::copy(simulation.grid().begin(), simulation.grid().end(), result.mutable_data());
            return result;
        });
    // Keep the GIL for all methods: callers cannot concurrently mutate one instance.
}
