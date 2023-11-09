#include "utils.h"

bool CudaUtils::cudaDeviceExists(int device_id)
{
    int device_count = 0;
    // cudaError_t err = cudaGetDeviceCount(&device_count);
    // if (err != cudaSuccess) {
    //     // throw CudaError(err);
    //     return false;
    // }
    return (device_id >= 0 && device_id < device_count);
}