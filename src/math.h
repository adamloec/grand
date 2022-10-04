#ifndef TENSOR_INCL
#define TENSOR_INCL
    #include "tensor.h"
#endif

cudaError_t add(Grand::Tensor::Matrix c, Grand::Tensor::Matrix a, Grand::Tensor::Matrix b, int device);