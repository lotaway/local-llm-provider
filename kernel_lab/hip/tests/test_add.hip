#include <iostream>
#include <kernel_lab.hpp>

int main() {
  constexpr int N = 4;

  float hostA[N] = {1, 2, 3, 4};
  float hostB[N] = {10, 20, 30, 40};
  float hostC[N] = {0};

  float *devA;
  float *devB;
  float *devC;
  dim3 grid_size(5, 3, 1);

  CHECK_HIP_ERROR(hipMalloc(&devA, sizeof(hostA)));

  CHECK_HIP_ERROR(hipMalloc(&devB, sizeof(hostB)));

  CHECK_HIP_ERROR(hipMalloc(&devC, sizeof(hostC)));
  kernel_lab::launch_add(devA, devB, devC, N, 256);

  CHECK_HIP_ERROR(
      hipMemcpy(hostC, devC, sizeof(hostC), hipMemcpyDeviceToHost));

  std::cout << hostC[0] << std::endl;
}