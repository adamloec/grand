#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

#include "math.h"
#ifndef INCL_TENSOR
    #include "tensor.h"
#endif

// Thread block size
#define BLOCK_SIZE 2

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/