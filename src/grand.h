#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CORE_INCL
#define CORE_INCL
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    using namespace std;
#endif
#ifndef TENSOR_INCL
#define TENSOR_INCL
    #include "tensor.h"
#endif

// Thread block size
#define BLOCK_SIZE 2

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/