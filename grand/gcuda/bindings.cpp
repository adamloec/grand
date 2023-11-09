#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// #include "src/cuda_error.h"
#include "src/utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_gcuda, m)
{
    m.doc() = R"pbdoc(
        gcuda library.
        -----------------------

        .. currentmodule:: _gcuda

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    // py::register_exception<CudaError>(m, "CudaError");
    py::class_<CudaUtils>(m, "CudaUtils")
        .def_static("cudaDeviceExists", &CudaUtils::cudaDeviceExists);
}