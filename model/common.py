import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

import resnet

def create_classifier(name: str, n_classes: int, pretrained: bool=False) -> nn.Module: 
    match name: 
        case 'resnet20': 
            model_ = resnet.resnet20(n_classes)
        case 'resnet32': 
            model_ = resnet.resnet32(n_classes)
        case 'resnet44': 
            model_ = resnet.resnet44(n_classes)
        case 'resnet56': 
            model_ = resnet.resnet56(n_classes)
        case 'resnet110': 
            model_ = resnet.resnet110(n_classes)
        case _:
            raise ValueError(f'Unknown classifier: {name}')
    if pretrained:
        raise NotImplementedError('Pretrained models are not supported yet')
    return model_
