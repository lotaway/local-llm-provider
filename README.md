# Local LLM Provider

This is use for [for VSCode Continue Plugin](https://docs.continue.dev), to provider a local LLM to provide code completion and chat, agent(edit,apply) ability.

## Usage

Download the openai standard model from [Hugging Face](https://huggingface.co) and put it into the `model` directory.

Change the `model_name` in `model_provider.py` to the downloaded model name.

Run the `main.py` file to start the server.

Install Vscode Continue Plugin and connect to the server.

## AMD Required

### Linux/WSL

Need install ROCm and [torch ROCm version](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html) in Linux/WSL with AMD GPU.

### Window

Use WSL ROCm is the best, either need install [Zluda](https://github.com/vosen/ZLUDA.git) or install microsoft dml with `pip install torch-directml`

If choose zluda, need to install torch with cuda version, use `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`(util 2025/10/11, zluda not implemented cuda13.0 yep, 12.8 is the latest)

## About bitsandbytes

*Only can use in cuda or amd with linux+rocm, amd with window+zluda no match, zluda did not support bitsandbytes*

Official documents clearly state that AMD GPUs must be compiled from source:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
```

### for linux rocm

```base
# Switch to the version you installed, such as 0.47.0
git fetch origin rocm-build-update:rocm-build-update
git switch rocm-build-update

# Setup ROCm environment variables
export BNB_ROCM_ARCH=1
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export HIP_PATH=$ROCM_HOME

# compile
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a;gfx942;gfx1100" .
cmake --build .
```

If got error like:

```bash
CMake Error at /opt/rocm/lib/cmake/hsa-runtime64/hsa-runtime64Targets.cmake:80 (message):
  The imported target "hsa-runtime64::hsa-runtime64" references the file

     "/opt/rocm/lib/libhsa-runtime64.so.1.15.60400"

  but this file does not exist.
```

Follow [Issue: cmake hsa-runtime64 in wsl2 not correctly set the library](https://github.com/ROCm/ROCm/issues/3606) to fix it with:

```bash
cd /opt/rocm/lib/
# Notice: Those number behind libhsa-runtime64.so, first one is the required version, second one is the installed version.
ln -s libhsa-runtime64.so.1.14 libhsa-runtime64.so.1.15.60400
```

After build success, you will see:
```bash
[100%] Linking CXX shared library bitsandbytes/libbitsandbytes_rocm64.so
[100%] Built target bitsandbytes
```

Take `libbitsandbytes_rocm64.so` file to `/home/{user}/miniforge3/envs/python{version}/lib/python{version}/site-packages/bitsandbytes` directory.
Notice: I use miniforge3 as python environment here, if you use other python environment, you should change the directory accordingly. Such as if you're using `anaconda`, directory prefix are `/home/{user}/canda/`

Done all process, now you can use `bitsandbytes` in your python code.

## For Multiple Mac (Apple silicon) devices

You can use [exo](https://github.com/exo-explore/exo) to start the mac device as node, multiple devices as clusters, and let data flow between them to use both gpu.