#include <hip/hip_runtime.h>
#include <kernel_lab.hpp>
#include <iostream>

__global__ void add_kernel(
    float *a,
    float *b,
    float *c,
    int n)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

void checkHipError(hipError_t err)
{
    if (err != hipSuccess)
    {
        std::cerr
            << hipGetErrorString(err)
            << std::endl;

        std::exit(1);
    }
}

int main()
{
    constexpr int N = 4;

    float hostA[N] = {1, 2, 3, 4};
    float hostB[N] = {10, 20, 30, 40};
    float hostC[N] = {0};

    float *devA;
    float *devB;
    float *devC;

    checkHipError(
        hipMalloc(&devA, sizeof(hostA)));

    checkHipError(
        hipMalloc(&devB, sizeof(hostB)));

    checkHipError(
        hipMalloc(&devC, sizeof(hostC)));

    checkHipError(
        hipMemcpy(
            devA,
            hostA,
            sizeof(hostA),
            hipMemcpyHostToDevice));

    checkHipError(
        hipMemcpy(
            devB,
            hostB,
            sizeof(hostB),
            hipMemcpyHostToDevice));

    add_kernel<<<1, 4>>>(
        devA,
        devB,
        devC,
        N);

    checkHipError(
        hipDeviceSynchronize());

    checkHipError(
        hipMemcpy(
            hostC,
            devC,
            sizeof(hostC),
            hipMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        std::cout << hostC[i] << " ";

    std::cout << std::endl;

    hipFree(devA);
    hipFree(devB);
    hipFree(devC);
}