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
#ifndef MATH_INCL
#define MATH_INCL
    #include "math.h"
#endif

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/