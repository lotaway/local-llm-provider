#include <hip/hip_runtime.h>
#include <iostream>
#include <kernel_lab.hpp>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

void kernel_lab::checkHipError(hipError_t err, const char fileName[], int line) {
  if (err != hipSuccess) {
    std::cerr << fileName << line << hipGetErrorString(err) << std::endl;
    __FILE__;
    std::exit(1);
  }
}

void kernel_lab::performance(hipEvent_t start, hipEvent_t stop) {
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventRecord(stop));
    float duration;
    CHECK_HIP_ERROR(hipEventElapsedTime(&duration, start, stop));
    std::cout << "performance duration time:" << duration << "ms" << std::endl;
    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
}

void kernel_lab::launch_add(float *a, float *b, float *c, int n, int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  add_kernel<<<grid_size, block_size>>>(a, b, c, n);
  CHECK_HIP_ERROR(hipDeviceSynchronize());
}