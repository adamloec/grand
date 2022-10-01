#include "utils.h"

__global__ void addKernal(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

cudaError_t add(int *c, const int *a, const int *b, unsigned int size, int device=0)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    Error:
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        
        return cudaStatus;
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = add(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Addition failed.");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}