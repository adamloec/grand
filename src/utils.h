#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Thread block size
#define BLOCK_SIZE 16

// Tensor struct
// width: width of matrix
// height: height of matrix
// data: 2d matrix
typedef struct
{
    int width;
    int height;
    float *data;
} Tensor;



cudaError_t add(int *c, const int *a, const int *b, unsigned int size, int device);