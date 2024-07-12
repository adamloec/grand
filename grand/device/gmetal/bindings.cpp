#include <pybind11/pybind11.h>
#include "src/example.h"

namespace py = pybind11;

PYBIND11_MODULE(_gmetal, m) {
    m.doc() = "Python bindings for example";
    m.def("add", &add, "Add two integers");
}