#include <iostream>

template <typename T>
__global__ void addKernel(T* a, T* b, T* c, int width, int height)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col < width*height)
    {
        c[col] = a[col] + b[col]
    }
}

template <typename T>
__global__ void mulKernel(T* a, T* b, T* c, int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0;
}

template <typename T>
extern "C" void 