#ifndef MATH_INCL
#define MATH_INCL
    #include <math.h>
    #include "math.h"
#endif

// ===================================================================================================
// NVIDIA COMPUTE CAPABILITY 8.6 SUPPORTED
// https://en.wikipedia.org/wiki/CUDA
//
// MAXIMUMS
// Threads per block = 1024
// Grids = 128
// Grid dimensions = (x, y, z)
//
//
// EXAMPLE KERNEL CALL
// kernel<<<ceil(n/256), 256>>>(args); //// <<<BLOCKS, THREADS PER BLOCK>>> n = flattened size of tensor
//
// EXAMPLE ERROR CHECKING
// if (err != cudaSuccess)
// { 
//    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
// }
//
// ===================================================================================================

namespace Grand
{
    // Add 2 tensor's kernel function.
    //
    // Tensor::Matrix c = Output tensor
    // Tensor::Matrix a/b = Input tensor's
    __global__ void addKernel(Tensor::Tensor c, Tensor::Tensor a, Tensor::Tensor b)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < a.width*a.height)
        {
            c.data[i] = a.data[i] + b.data[i];
        }
    }

    // Dot product 2 tensor's kernel function.
    //
    // m * n matrix
    // n * k matrix
    // Tensor::Matrix c = Output tensor
    // Tensor::Matrix a/b = Input tensor's
    __global__ void dotKernel(Tensor::Tensor c, Tensor::Tensor a, Tensor::Tensor b)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < a.width*a.height)
        {
            c.data[i] = a.data[i] + b.data[i];
        }
    }

    // Add 2 tensor's function.
    //
    // Tensor::Array c = Output tensor
    // Tensor::Array a/b = Input tensor's (m * n)
    cudaError_t add(Tensor::Tensor c, Tensor::Tensor a, Tensor::Tensor b, int device=0)
    {
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;
        size_t size;
        cudaError_t cudaStatus;

        // CUDA device check
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
            goto Error;
        }

        // Tensor input dimensions equality check
        if (a.width != b.width && a.height != b.height && c.width != a.width && c.height != a.height)
        {
            fprintf(stderr, "ERROR: Tensor dimensions do not match. A: {%d, %d} B: {%d, %d} C: {%d, %d}\n", a.width, a.height, b.width, b.height, c.width, c.height);
            goto Error;
        }

        // Data size (bytes)
        size = a.width * a.height * sizeof(float);

        // Initialize device tensor's width and height
        dev_a.width = a.width;
        dev_a.height = a.height;
        dev_b.width = b.width;
        dev_b.height = b.height;
        dev_c.width = c.width;
        dev_c.height = c.height;
        
        // Device memory allocation for input tensors
        cudaMalloc(&dev_a.data, size);
        cudaMalloc(&dev_b.data, size);

        // Copy input tensor's from host to device memory
        cudaMemcpy(dev_a.data, a.data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b.data, b.data, size, cudaMemcpyHostToDevice);

        // Device memory allocation for output tensor
        cudaMalloc(&dev_c.data, size);

        // Invoke kernel with specified kernel dimensions
        addKernel<<<ceil((a.width*a.height)/256.0), 256>>>(dev_c, dev_a, dev_b);

        // Kernel synchronize, checks for kernel errors
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
            goto Error;
        }

        // Copy output tensor from device to host memory
        cudaStatus = cudaMemcpy(c.data, dev_c.data, size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: CUDAMEMCPY: %d\n", cudaStatus);
            goto Error;
        }

    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);

        return cudaStatus;
    }

    // Multiply 2 tensor's function.
    //
    // Tensor::Array c = m * k output tensor
    // Tensor::Array a = m * n input tensor
    // Tensor::Array b = n * k input tensor
    cudaError_t dot(Tensor::Tensor c, Tensor::Tensor a, Tensor::Tensor b, int device=0)
    {
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;
        size_t size;
        cudaError_t cudaStatus;

        // CUDA device check
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
            goto Error;
        }

        

    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);

        return cudaStatus;
    }
}