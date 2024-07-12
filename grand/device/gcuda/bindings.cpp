#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "src/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_gcuda, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<size_t>())
        .def("copy_from_host", [](Tensor &self, py::array_t<float> data) {
            py::buffer_info info = data.request();
            self.copy_from_host(static_cast<float*>(info.ptr), info.size * sizeof(float));
        })
        .def("get_ptr", &Tensor::get_ptr);
}