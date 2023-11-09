#ifndef UTILS
#define UTILS

#include <cuda_runtime.h>
#include <stdexcept>

// #include "cuda_error.h"

class CudaUtils
{
    public:
        static bool cudaDeviceExists(int device_id);
};

#endif