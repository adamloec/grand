#include "bindings.h"

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

    py::register_exception<CudaError>(m, "CudaError");
    
    m.def("cudaDeviceExists", &cudaDeviceExists, R"pbdoc(
        Checks if a specific CUDA-enabled device exists based on the given device ID.

        Parameters:
            device_id (int): The ID of the device to check.

        Returns:
            bool: True if the device exists, False otherwise.

        Throws:
            CudaError: If there is an error in retrieving device information.

        Examples:
            >>> from . import _gcuda 
            >>> _gcuda.cudaDeviceExists(0) # Checks if the first CUDA device is available
            True
    )pbdoc");
}