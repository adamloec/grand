#include "utils.h"

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/

__global__ void addKernel(Tensor c, Tensor a, Tensor b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    c.data[i][j] = a.data[i][j] + b.data[i][j];
}

cudaError_t add(Tensor c, Tensor a, Tensor b, int device=0)
{
    Tensor dev_a;
    Tensor dev_b;
    Tensor dev_c;
    size_t size;
    cudaError_t cudaStatus;

    // CUDA device check
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
    }

    // Tensor input dimensions equality check
    if (a.width != b.width && a.height != b.height && c.width != a.width && c.height != a.height)
    {
        fprintf(stderr, "ERROR: Tensor dimensions do not match. A: {%d, %d} B: {%d, %d} C: {%d, %d}\n", a.width, a.height, b.width, b.height, c.width, c.height);
    }

    // Constant width/height dimensions for Tensors
    size = a.width * a.height * sizeof(float);

    // Initialize input tensors and copy to memory
    cudaMalloc(&dev_a.data, size);
    cudaMemcpy(dev_a.data, a.data, size, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_b.data, size);
    cudaMemcpy(dev_b.data, b.data, size, cudaMemcpyHostToDevice);

    // Initialize output tensor and copy to memory
    cudaMalloc(&dev_c.data, size);

    // Generate kernel dimensions, invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b.width / dimBlock.x, a.height / dimBlock.y);
    addKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);

    // Kernel synchronize, checks for kernel errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
    }

    // Read output tensor from memory
    cudaMemcpy(c.data, dev_c.data, size, cudaMemcpyDeviceToHost);

    return cudaStatus;
}

int main()
{
    Tensor c;
    Tensor a;
    Tensor b;

    // Test data
    c.height = 2;
    c.width = 2;
    c.data = (float**)malloc(2*sizeof(float));
    for (int i = 0; i < 2; i++)
        c.data[i] = (float*)malloc(2 * sizeof(float));

    a.height = 2;
    a.width = 2;
    a.data = (float**)malloc(2*sizeof(float));
    for (int i = 0; i < 2; i++)
        a.data[i] = (float*)malloc(2 * sizeof(float));
    a.data[0][0] = 1.0;
    a.data[0][1] = 2.0;
    a.data[1][0] = 3.0;
    a.data[1][1] = 4.0;

    b.height = 2;
    b.width = 2;
    b.data = (float**)malloc(2*sizeof(float));
    for (int i = 0; i < 2; i++)
        b.data[i] = (float*)malloc(2 * sizeof(float));
    b.data[0][0] = 1.0;
    b.data[0][1] = 2.0;
    b.data[1][0] = 3.0;
    b.data[1][1] = 4.0;

    // Add vectors in parallel.
    cudaError_t cudaStatus = add(c, a, b);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    // Output
    fprintf(stderr, "True ");
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        fprintf(stderr, "%f ", c.data[i][j]);


    free(c.data);
    free(a.data);
    free(b.data);

    cudaDeviceReset();

    return 0;
}