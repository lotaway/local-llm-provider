def test_cuda():
    import sys
    import os
    print(os.path.dirname(sys.executable))
    print(sys.exec_prefix)
    import ctypes.util
    print(ctypes.util.find_library('nvcuda'))
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))

if __name__ == '__main__':
    test_cuda()