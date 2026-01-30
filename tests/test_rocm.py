def test_ROCm():
    import torch

    print(f"amd hip version: {torch.version.hip}")  # Will have hip version '6.4.2'
    print(f"gpu cuda is available: {torch.cuda.is_available()}")
    print(
        f"gpu device: {torch.cuda.current_device()}"
    )  # Must be have device index such as '0', or can't run in gpu


if __name__ == "__main__":
    test_ROCm()
