import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from model import resnet

def create_classifier(name: str, n_classes: int, pretrained: bool=False) -> nn.Module: 
    if name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        model_ = getattr(resnet, name)(n_classes)
    else: 
        raise ValueError(f'Unknown classifier: {name}')
    if pretrained:
        raise NotImplementedError('Pretrained models are not supported yet')
    return model_
