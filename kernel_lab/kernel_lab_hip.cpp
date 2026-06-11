#include <hip/hip_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

__global__ void add_kernel(
    float* a,
    float* b,
    float* c,
    int n
) {
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void checkHipError(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr
            << "HIP Error: "
            << hipGetErrorString(err)
            << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

// for test
// int main()
// {
//     int host[4] = {1, 2, 3, 4};

//     int* device;

//     checkHipError(hipMalloc(&device, sizeof(host)));
//     hipMemcpy(
//         device,
//         host,
//         sizeof(host),
//         hipMemcpyHostToDevice
//     );

//     add<<<1, 4>>>(device);

//     hipDeviceSynchronize();

//     checkHipError(hipMemcpy(
//         host,
//         device,
//         sizeof(host),
//         hipMemcpyDeviceToHost
//     ));

//     std::cout
//         << host[0] << " "
//         << host[1] << " "
//         << host[2] << " "
//         << host[3]
//         << std::endl;

//     hipFree(device);
// }

// kernel<<<blocks, threads>>>();

// checkHip(
//     hipGetLastError()
// );

// checkHip(
//     hipDeviceSynchronize()
// );

// for hip test
torch::Tensor add(
    torch::Tensor a,
    torch::Tensor b
) {
    auto c = torch::zeros_like(a);

    int n = a.numel();

    add_kernel<<<
        (n + 255) / 256,
        256
    >>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}

PYBIND11_MODULE(kernel_lab_hip, m) {
    m.def("add", &add);
}