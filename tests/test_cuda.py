def test_cuda():
    import sys
    import os

    print(os.path.dirname(sys.executable))
    print(sys.exec_prefix)
    import ctypes.util

    print(f'nvcuda lib path: {ctypes.util.find_library("nvcuda")}')
    import torch

    is_cuda = torch.cuda.is_available()
    print(f"cuda available: {is_cuda}")
    print(f"cuda version: {torch.version.cuda}")
    if is_cuda:
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    else:
        print("No cuda device available")


if __name__ == "__main__":
    test_cuda()
