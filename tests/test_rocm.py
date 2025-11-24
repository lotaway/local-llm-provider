def test_ROCm():
    import torch
    print(torch.version.hip)   # Will have hip version '6.4.2'
    print(torch.cuda.current_device()) # Must be have device index such as '0', or can't run in gpu

if __name__ == '__main__':
    test_ROCm()

