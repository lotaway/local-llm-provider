#include <kernel_lab.cup>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  auto c = torch::zeros_like(a);

  int n = a.numel();

  kernel_lab::launch_add(a.data_ptr<float>(), b.data_ptr<float>(),
                         c.data_ptr<float>(), n, 256);

  cudaDeviceSynchronize();

  return c;
}

PYBIND11_MODULE(kernel_lab, m) { m.def("add", &add); }