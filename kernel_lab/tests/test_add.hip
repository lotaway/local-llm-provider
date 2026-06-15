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

  kernel_lab::checkHipError(hipMalloc(&devA, sizeof(hostA)));

  kernel_lab::checkHipError(hipMalloc(&devB, sizeof(hostB)));

  kernel_lab::checkHipError(hipMalloc(&devC, sizeof(hostC)));
  kernel_lab::launch_add(devA, devB, devC, N, 256);

  kernel_lab::checkHipError(
      hipMemcpy(hostC, devC, sizeof(hostC), hipMemcpyDeviceToHost));

  std::cout << hostC[0] << std::endl;
}