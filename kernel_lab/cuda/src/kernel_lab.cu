#include <cuda_runtime.h>
#include <iostream>
#include <kernel_lab.cup>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

void kernel_lab::checkHipError(cudaError_t err, const char fileName[], int line) {
  if (err != cudaSuccess) {
    std::cerr << fileName << line << cudaGetErrorString(err) << std::endl;
    __FILE__;
    std::exit(1);
  }
}

void kernel_lab::performance(cudaEvent_t start, cudaEvent_t stop) {
    kernel_lab::checkHipError(cudaEventCreate(&start), __FILE__, __LINE__);
    kernel_lab::checkHipError(cudaEventRecord(stop), __FILE__, __LINE__);
    float duration;
    kernel_lab::checkHipError(cudaEventElapsedTime(&duration, start, stop), __FILE__, __LINE__);
    std::cout << "performance duration time:" << duration << "ms" << std::endl;
    kernel_lab::checkHipError(cudaEventDestroy(start), __FILE__, __LINE__);
    kernel_lab::checkHipError(cudaEventDestroy(stop), __FILE__, __LINE__);
}

void kernel_lab::launch_add(float *a, float *b, float *c, int n, int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  add_kernel<<<grid_size, block_size>>>(a, b, c, n);
  kernel_lab::checkHipError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}