#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t add(int *c, const int *a, const int *b, unsigned int size, int device);