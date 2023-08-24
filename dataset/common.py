import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100, ImageNet, VOCSegmentation

def n_classes(name: str): 
    if name == 'cifar10': 
        return 10
    elif name == 'cifar100': 
        return 100
    elif name == 'imagenet':
        return 1000
    raise ValueError(f'Unknown dataset: {name}')

def create_dataset(name: str, train: bool, transform=None): 
    root = f'dataset/{name}'
    if name == 'cifar10': 
        return CIFAR10(root=root, train=train, transform=transform, download=False)
    elif name == 'cifar100': 
        return CIFAR100(root=root, train=train, transform=transform, download=False)
    elif name == 'imagenet':
        return ImageNet(root=root, split='train' if train else 'val', transform=transform)
    elif name == 'voc': 
        return VOCSegmentation(root=root, year='2012', image_set='train' if train else 'val', transform=transform)
    raise ValueError(f'Unknown dataset: {name}')
