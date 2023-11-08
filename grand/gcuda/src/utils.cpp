#include "utils.h"

bool cudaDeviceExists(int device_id)
{
    int device_count = 0;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        
    }
    return (device_id >= 0 && device_id < device_count);
}