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
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (col < a.width*a.height)
        {
            c.data[col] = a.data[col] + b.data[col];
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
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;

        float sum = 0.0;

        if (col < b.height && row < a.width)
        {
            for (int i = 0; i < a.height; i++)
            {
                sum += a.data[row * a.height + i] * b.data[i * b.height + col];
            }
            c.data[row * b.height + col] = sum;
        }
    }

    // Add 2 tensor's function.
    //
    // Tensor::Array c = Output tensor
    // Tensor::Array a/b = Input tensor's (m * n)
    Tensor::Tensor add(Tensor::Tensor a, Tensor::Tensor b, int device=0)
    {
        // Create device tensors.
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;

        // Output tensor.
        Tensor::Zeros c(a);

        // Data size (bytes). Same size for all tensors for proper addition.
        size_t size;

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

        // Initialize device tensor's width and height and size.
        dev_a.width = a.width;
        dev_a.height = a.height;
        dev_b.width = b.width;
        dev_b.height = b.height;
        dev_c.width = c.tensor.width;
        dev_c.height = c.tensor.height;
        size = a.width * a.height * sizeof(float);
        
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

        return c.tensor;

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
        // Create device tensors.
        Tensor::Tensor dev_a;
        Tensor::Tensor dev_b;
        Tensor::Tensor dev_c;

        // Output tensor.
        Tensor::Zeros c(a.width, b.height);

        // Data size (bytes).
        size_t size;

        // Grid/block dim3
        int grid_rows = (a.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_cols = (b.height + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

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

        // Initialize device tensor's width and height and size.
        dev_a.width = a.width;
        dev_a.height = a.height;
        dev_b.width = b.width;
        dev_b.height = b.height;
        dev_c.width = c.tensor.width;
        dev_c.height = c.tensor.height;
        size = a.width * a.height * sizeof(float);

        // Device memory allocation for input tensors.
        cudaMalloc(&dev_a.data, size);
        cudaMalloc(&dev_b.data, size);

        // Copy input tensor's from host to device memory.
        cudaMemcpy(dev_a.data, a.data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b.data, b.data, size, cudaMemcpyHostToDevice);

        // Device memory allocation for output tensor.
        cudaMalloc(&dev_c.data, size);

        // Invoke kernel with specified kernel dimensions.
        dotKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);

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

        return c.tensor;

    Error:
        cudaFree(dev_c.data);
        cudaFree(dev_a.data);
        cudaFree(dev_b.data);

        c.tensor.status = 0;
        return c.tensor;
    }
}