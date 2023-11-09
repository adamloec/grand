#ifndef CUDA_ERROR
#define CUDA_ERROR

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

class CudaError : public std::runtime_error
{
    public:
        explicit CudaError(cudaError_t error)
        : std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))) {}
};

#endif