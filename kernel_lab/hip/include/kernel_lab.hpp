#pragma once

#include <hip/hip_runtime.h>

#define CHECK_HIP_ERROR(...) kernel_lab::checkHipError(__VA_ARGS__, __FILE__, __LINE__)

namespace kernel_lab {

void launch_add(float *a, float *b, float *c, int n, int block_size);

// void checkHipError(hipError_t);

void checkHipError(hipError_t, const char[], int);

void performance(hipEvent_t, hipEvent_t);

} // namespace kernel_lab