#include <iostream>
#include <kernel_lab.cuh>

int main() {
  constexpr int N = 4;

  float hostA[N] = {1, 2, 3, 4};
  float hostB[N] = {10, 20, 30, 40};
  float hostC[N] = {0};

  float *devA;
  float *devB;
  float *devC;
  dim3 grid_size(5, 3, 1);

  CHECK_CUDA_ERROR(cudaMalloc(&devA, sizeof(hostA)), __FILE__, __LINE__);

  CHECK_CUDA_ERROR(cudaMalloc(&devB, sizeof(hostB)), __FILE__, __LINE__);

  CHECK_CUDA_ERROR(cudaMalloc(&devC, sizeof(hostC)), __FILE__, __LINE__);
  kernel_lab::launch_add(devA, devB, devC, N, 256);

  CHECK_CUDA_ERROR(
      cudaMemcpy(hostC, devC, sizeof(hostC), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

  std::cout << hostC[0] << std::endl;
}