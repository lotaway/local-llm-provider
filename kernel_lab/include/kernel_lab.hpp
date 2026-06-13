#pragma once

#include <hip/hip_runtime.h>

namespace kernel_lab {

void launch_add(float *a, float *b, float *c, int n, int block_size);

void checkHipError(hipError_t err);

} // namespace kernel_lab