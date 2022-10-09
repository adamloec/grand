#ifndef CORE_INCL
#define CORE_INCL
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    using namespace std;
#endif
#ifndef TENSOR_INCL
#define TENSOR_INCL
    #include "tensor.h"
#endif

namespace Grand
{
    // Add 2 tensor's function.
    //
    // Tensor::Matrix c = Output tensor
    // Tensor::Matrix a/b = Input tensor's
    cudaError_t add(Tensor::Tensor c, Tensor::Tensor a, Tensor::Tensor b, int device);
}