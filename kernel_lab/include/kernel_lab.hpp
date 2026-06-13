#pragma once

#include <hip/hip_runtime.h>

namespace kernel_lab {
    
    __global__ void add_kernel(float *a, float *b, float *c, int n);

}