import torch
import torchvision
from torchvision import transforms
from einops import rearrange

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root, train=train, transform=transform, download=download)
    
    def __getitem__(self, idx):
        imgs, label = super().__getitem__(idx)
        assert imgs.shape == (3, 32, 32)
        imgs = rearrange(imgs, 'c h w -> h w c')
        return {"image": imgs, "label": label}


def CIFAR10_Train(root='/SSD'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return CIFAR10(root=root, train=True, download=True, transform=transform_train)


def CIFAR10_Validation(root='/SSD'):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return CIFAR10(root=root, train=False, download=True, transform=transform_val)

if __name__ == '__main__':
    CIFAR10_Train()