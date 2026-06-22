# kernel_lab

Dual-platform GPU kernel experiments (CUDA + HIP), exposed as a Python module via pybind11.

## CUDA → HIP Conversion with HIPIFY

This project uses AMD ROCm [HIPIFY](https://github.com/ROCm/HIPIFY) to translate CUDA sources to HIP.

```bash
# Convert the entire cuda/ directory to hip/
hipify-clang \
  -I cuda/include \
  -I $(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/../include \
  --cuda-path=/usr/local/cuda \
  -o hip \
  cuda/src/*.cu

# Note: after conversion, manually update CMakeLists.txt:
#   project(kernel_lab LANGUAGES CXX CUDA) → project(kernel_lab LANGUAGES CXX HIP)
#   src/kernel_lab.cu → src/kernel_lab.hip
```

### Options

| Flag | Description |
|------|-------------|
| `-I <dir>` | Add include search path |
| `--cuda-path=<dir>` | CUDA installation path |
| `-o <dir>` | Output directory (mirrors source tree structure) |
| `--roc` | Translate to roc instead of hip where possible |

## Build

```bash
cd kernel_lab/cuda && bash install.sh   # CUDA
cd kernel_lab/hip && bash install.sh    # HIP
```

## Directory Layout

```
kernel_lab/
├── cuda/       # CUDA implementation (.cu / .cuh)
├── hip/        # HIP implementation (.hip / .hpp)
├── docker-compose.dev.yml
├── Dockerfile_DevEnv
└── requirement.txt
```
