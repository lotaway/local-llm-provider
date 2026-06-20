#pragma once

#include <cuda_runtime.h>

namespace kernel_lab {

void launch_add(float *a, float *b, float *c, int n, int block_size);

// void checkHipError(cudaError_t);

void checkHipError(cudaError_t, const char[], int);

void performance(cudaEvent_t, cudaEvent_t);

} // namespace kernel_lab