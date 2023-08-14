import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100, ImageNet

Cifar10 = CIFAR10
Cifar100 = CIFAR100

def dataset(name: str, train: bool): 
    root = f'dataset/{name}'
    if name == 'cifar10': 
        return Cifar10(root=root, train=train, download=False)
    elif name == 'cifar100': 
        return Cifar100(root=root, train=train, download=False)
    elif name == 'imagenet':
        raise NotImplementedError('ImageNet is not implemented yet')
    raise ValueError(f'Unknown dataset: {name}')
