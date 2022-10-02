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
        fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
        goto Error;
    }

    // Allocate mem on device
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy input matrixes to memory
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    addKernal<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for kernel errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Kernel synchronize, checks for kernel errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
        goto Error;
    }

    // Copy output matrix to memory
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

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
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaDeviceReset();

    return 0;
}