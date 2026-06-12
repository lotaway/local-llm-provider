#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "kernel_lab.hpp"

torch::Tensor add(
    torch::Tensor a,
    torch::Tensor b
) {
    auto c = torch::zeros_like(a);

    int n = a.numel();

    kernel_lab::add_kernel<<<
        (n + 255) / 256,
        256
    >>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    hipDeviceSynchronize();

    return c;
}

PYBIND11_MODULE(kernel_lab, m) {
    m.def("add", &add);
}