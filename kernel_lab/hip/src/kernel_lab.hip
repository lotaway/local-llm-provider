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
    kernel_lab::checkHipError(hipEventCreate(&start), __FILE__, __LINE__);
    kernel_lab::checkHipError(hipEventRecord(stop), __FILE__, __LINE__);
    float duration;
    kernel_lab::checkHipError(hipEventElapsedTime(&duration, start, stop), __FILE__, __LINE__);
    std::cout << "performance duration time:" << duration << "ms" << std::endl;
    kernel_lab::checkHipError(hipEventDestroy(start), __FILE__, __LINE__);
    kernel_lab::checkHipError(hipEventDestroy(stop), __FILE__, __LINE__);
}

void kernel_lab::launch_add(float *a, float *b, float *c, int n, int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  add_kernel<<<grid_size, block_size>>>(a, b, c, n);
  kernel_lab::checkHipError(hipDeviceSynchronize(), __FILE__, __LINE__);
}