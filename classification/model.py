import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

class Classifier(nn.Module): 
    def __init__(self, backbone: str, n_classes: int, pretrained: bool=True):
        super().__init__()
        self.backbone = create_model(backbone)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)
        self.n_classes = n_classes
    
    def forward(self, x): 
        return self.backbone(x)

def create_model(name: str, pretrained: bool=True): 
    match name: 
        case 'resnet34': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
        case 'resnet50': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        case 'resnet101':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained)
        case 'resnet152': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=pretrained)
        case 'vgg11': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=pretrained)
        case 'vgg11_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=pretrained)
        case 'vgg13': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=pretrained)
        case 'vgg13_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=pretrained)
        case 'vgg16': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=pretrained)
        case 'vgg16_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=pretrained)
        case 'vgg19': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=pretrained)
        case 'vgg19_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=pretrained)
        case 'alexnet': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)
        case 'googlenet': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=pretrained)
        case _: 
            raise ValueError(f'Unknown model: {name}')
    return model
