#include "utils.h"

bool cudaDeviceExists(int device_id)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw CudaError(err);
    }
    return (device_id >= 0 && device_id < device_count);
}