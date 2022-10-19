// ===================================================================================================
// Author: Adam Loeckle
// Date: 10/10/2022
// Description: Math source file.
// ===================================================================================================

#ifndef MATH_INCL
#define MATH_INCL
    #include <math.h>
    #include "math.h"
#endif

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
    Tensor::Tensor add(Tensor::Tensor a, Tensor::Tensor b, int device=0)
    {
        // CUDA device check.
        cudaError_t cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
            goto Error;
        }

        // Tensor input dimensions equality check.
        if (a.width != b.width && a.height != b.height)
        {
            fprintf(stderr, "ERROR: Tensor dimensions do not match for addition. A: {%d, %d} B: {%d, %d}\n", a.width, a.height, b.width, b.height);
            goto Error;
        }

        // Create device tensors.
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;

        // Output tensor.
        Tensor::Zeros c(a);

        // Initialize device tensor's width and height.
        dev_a.width = a.width;
        dev_a.height = a.height;
        dev_b.width = b.width;
        dev_b.height = b.height;
        dev_c.width = c.tensor.width;
        dev_c.height = c.tensor.height;

        // Data size (bytes). Same size for all tensors for proper addition.
        size_t size = a.width * a.height * sizeof(float);
        
        // Device memory allocation for input tensors.
        cudaMalloc(&dev_a.data, size);
        cudaMalloc(&dev_b.data, size);

        // Copy input tensor's from host to device memory.
        cudaMemcpy(dev_a.data, a.data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b.data, b.data, size, cudaMemcpyHostToDevice);

        // Device memory allocation for output tensor.
        cudaMalloc(&dev_c.data, size);

        // Invoke kernel with specified kernel dimensions.
        addKernel<<<ceil((a.width*a.height)/256.0), 256>>>(dev_c, dev_a, dev_b);

        // Kernel synchronize, checks for kernel errors.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
            goto Error;
        }

        // Copy output tensor from device to host memory.
        cudaStatus = cudaMemcpy(c.tensor.data, dev_c.data, size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: CUDAMEMCPY: %d\n", cudaStatus);
            goto Error;
        }

    // Error checking.
    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);

        c.tensor.status = 0;
        return c.tensor;
    }

    // Multiply 2 tensor's function.
    //
    // Tensor::Array c = m * k output tensor
    // Tensor::Array a = m * n input tensor
    // Tensor::Array b = n * k input tensor
    Tensor::Tensor dot(Tensor::Tensor a, Tensor::Tensor b, int device=0)
    {
        // CUDA device check
        cudaError_t cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(stderr, "ERROR: Cuda enabled device {Device: %d} not found.\n", device);
            goto Error;
        }

        // Tensor input dimensions equality check.
        // a.height == b.width, c.width == a.width, c.height = b.height
        if (a.height != b.width)
        {
            fprintf(stderr, "ERROR: Tensor dimensions do not match for dot product. A: {%d, %d} B: {%d, %d}\n", a.width, a.height, b.width, b.height);
            goto Error;
        }

        // Create device tensors.
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;

        // Output tensor.
        Tensor::Zeros c(a.width, b.height);

        // Initialize device tensor's width and height.
        dev_a.width = a.width;
        dev_a.height = a.height;
        dev_b.width = b.width;
        dev_b.height = b.height;
        dev_c.width = c.tensor.width;
        dev_c.height = c.tensor.height;

        // Data size (bytes).
        size_t size;


    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);

        c.tensor.status = 0;
        return c.tensor;
    }
}