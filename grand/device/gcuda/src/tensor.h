#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

class Tensor {
public:
    Tensor(size_t size);
    ~Tensor();
    void copy_from_host(const float* host_data, size_t size);
    void* get_ptr();

private:
    void* d_ptr;
    size_t size;
};