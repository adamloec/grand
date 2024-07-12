#include "tensor.h"

Tensor::Tensor(size_t size) {
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    this->size = size;
}

Tensor::~Tensor() {
    cudaFree(d_ptr);
}

void Tensor::copy_from_host(const float* host_data, size_t size) {
    if (size != this->size) {
        throw std::runtime_error("Size mismatch between host data and device buffer");
    }
    cudaError_t err = cudaMemcpy(d_ptr, host_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from host to device");
    }
}

void* Tensor::get_ptr() {
    return d_ptr;
}