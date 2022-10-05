#ifndef MATH_INCL
#define MATH_INCL
    #include "math.h"
#endif
using namespace Grand;

// Thread block size
#define BLOCK_SIZE 2

__global__ void addKernel(Tensor::Matrix c, Tensor::Matrix a, Tensor::Matrix b)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    printf("ROW: %d\n", row);
    printf("COL: %d\n", col);
    printf("A: %f B: %f\n", a.tensor[row], b.tensor[row]);
    
    if (row < b.width && col < a.height)
        c.tensor[row] = a.tensor[row] + b.tensor[row];
}

cudaError_t add(Tensor::Matrix c, Tensor::Matrix a, Tensor::Matrix b, int device=0)
{
    Tensor::Matrix dev_a;
    Tensor::Matrix dev_b;
    Tensor::Matrix dev_c;
    size_t size;
    cudaError_t cudaStatus;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b.width / dimBlock.x, a.height / dimBlock.y);

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

    // Initialize input tensors and copy to memory
    dev_a.width = a.width;
    dev_a.height = a.height;
    size = a.width * a.height * sizeof(float);
    cudaMalloc(&dev_a.tensor, size);
    cudaMemcpy(dev_a.tensor, a.tensor, size, cudaMemcpyHostToDevice);

    dev_b.width = b.width;
    dev_b.height = b.height;
    size = b.width * b.height * sizeof(float);
    cudaMalloc(&dev_b.tensor, size);
    cudaMemcpy(dev_b.tensor, b.tensor, size, cudaMemcpyHostToDevice);

    // Initialize output tensor and copy to memory
    dev_c.width = c.width;
    dev_c.height = c.height;
    size = c.width * c.height * sizeof(float);
    cudaMalloc(&dev_c.tensor, size);

    // Generate kernel dimensions, invoke kernel
    addKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);

    // Kernel synchronize, checks for kernel errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Kernel synchronize failed: %d\n", cudaStatus);
        goto Error;
    }

    // Read output tensor from memory
    //cudaStatus = cudaMemcpy(c.tensor, dev_c.tensor, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: CUDAMEMCPY: %d\n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(dev_c.tensor);
    cudaFree(dev_a.tensor);
    cudaFree(dev_b.tensor);

    return cudaStatus;
}

int main()
{
    vector<vector<float>> data{{1, 2}, {3, 4}};
    Grand::Tensor::Matrix a(data);
    Grand::Tensor::Matrix b(data);
    Grand::Tensor::Matrix c;
    c.height = 2;
    c.width = 2;

    // Add vectors in parallel.
    cudaError_t cudaStatus = add(c, a, b);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    // Output
    // for (int i = 0; i < c.width*c.height; i++)
    // {
    //     cout << "C: " << c.tensor[i];
    //     cout << endl;
    // }

    free(c.tensor);
    free(a.tensor);
    free(b.tensor);

    cudaDeviceReset();

    return 0;
}