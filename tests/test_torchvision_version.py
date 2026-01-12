import torchvision, os

print(torchvision.__file__)
print(os.listdir(os.path.dirname(torchvision.__file__)))
