// ===================================================================================================
// Author: Adam Loeckle
// Date: 10/10/2022
// Description: Math header file.
// ===================================================================================================

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

// ===================================================================================================
// NVIDIA COMPUTE CAPABILITY 8.6 SUPPORTED
// https://en.wikipedia.org/wiki/CUDA
//
// MAXIMUMS
// Threads per block = 1024
// Grids = 128
// Grid dimensions = (x, y, z)
//
//
// EXAMPLE KERNEL CALL
// kernel<<<ceil(n/256), 256>>>(args); //// <<<BLOCKS, THREADS PER BLOCK>>> n = flattened size of tensor
//
// EXAMPLE ERROR CHECKING
// if (err != cudaSuccess)
// { 
//    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
// }
//
// ===================================================================================================

namespace Grand
{
    #define BLOCK_SIZE 16

    // Add 2 tensor's function.
    //
    // Tensor::Matrix c = Output tensor
    // Tensor::Matrix a/b = Input tensor's
    Tensor::Tensor add(Tensor::Tensor a, Tensor::Tensor b, int device);

    // Multiply 2 tensor's function.
    //
    // Tensor::Array c = m * k output tensor
    // Tensor::Array a = m * n input tensor
    // Tensor::Array b = n * k input tensor
    Tensor::Tensor dot(Tensor::Tensor a, Tensor::Tensor b, int device);
}