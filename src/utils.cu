#include "utils.h"

__global__ void addKernel(Tensor c, Tensor a, Tensor b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    
}

cudaError_t add(Tensor c, Tensor a, Tensor b, int device=0)
{
    Tensor dev_a;
    Tensor dev_b;
    Tensor dev_c;
    size_t size;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
        goto Error;
    }

    // Initialize input tensors and copy to memory
    dev_a.width = a.width; 
    dev_a.height = a.height;
    size = a.width * a.height * sizeof(float);
    cudaMalloc(&dev_a.data, size);
    cudaMemcpy(dev_a.data, a.data, size, cudaMemcpyHostToDevice);

    dev_b.width = b.width;
    dev_b.height = b.height;
    size = b.width * b.height * sizeof(float);
    cudaMalloc(&dev_b.data, size);
    cudaMemcpy(dev_b.data, b.data, size, cudaMemcpyHostToDevice);

    // Initialize output tensor and copy to memory
    dev_c.width = c.width; 
    dev_c.height = c.height;
    size = c.width * c.height * sizeof(float);
    cudaMalloc(&dev_c.data, size);

    // Generate kernel dimensions, invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b.width / dimBlock.x, a.height / dimBlock.y);
    addKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);

    // Kernel synchronize, checks for kernel errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
        goto Error;
    }

    // Read output tensor from memory
    cudaMemcpy(c.data, dev_c.data, size, cudaMemcpyDeviceToHost);

    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);
        
        return cudaStatus;
}

int main()
{
    // Add vectors in parallel.
    cudaError_t cudaStatus = add();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    cudaDeviceReset();

    return 0;
}