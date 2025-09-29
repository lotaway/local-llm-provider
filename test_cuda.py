import torch

def test_cuda() -> bool:
    return torch.cuda.is_available()

if __name__ == '__main__':
    print(test_cuda())