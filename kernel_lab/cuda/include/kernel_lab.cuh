#pragma once

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(...) kernel_lab::checkCudaError(__VA_ARGS__, __FILE__, __LINE__)

namespace kernel_lab {

void launch_add(float *a, float *b, float *c, int n, int block_size);

// void checkCudaError(cudaError_t);

void checkCudaError(cudaError_t, const char[], int);

void performance(cudaEvent_t, cudaEvent_t);

} // namespace kernel_lab