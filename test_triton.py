import torch
import triton
import triton.language as tl

def test():
    import types
    if not hasattr(triton.runtime, 'backend'):
        triton.runtime.backend = types.SimpleNamespace(CUDA=None, HIP=True)
        print("shim: added triton.runtime.backend.HIP")
    print(f"triton version: {triton.__version__}")
    print(f"triton cuda backend: {triton.runtime.backend.CUDA}")
    print(f"triton language: {tl}")

if __name__ == '__main__':
    test()