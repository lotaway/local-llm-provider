#include <hip/hip_runtime.h>
#include <iostream>
#include <kernel_lab.hpp>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

void kernel_lab::checkHipError(hipError_t err) {
  if (err != hipSuccess) {
    std::cerr << hipGetErrorString(err) << std::endl;

    std::exit(1);
  }
}

void kernel_lab::launch_add(float *a, float *b, float *c, int n, int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  add_kernel<<<grid_size, block_size>>>(a, b, c, n);
  checkHipError(hipDeviceSynchronize());
}