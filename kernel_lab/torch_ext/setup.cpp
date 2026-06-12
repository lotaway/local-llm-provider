#include <torch/extension.h>
#include <pybind11/pybind11.h>

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
    checkHipError(hipDeviceSynchronize());
    return c;
}

PYBIND11_MODULE(kernel_lab_hip, m) {
    m.def("add", &add);
}