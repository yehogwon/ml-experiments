import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

def create_model(name: str): 
    match name: 
        case 'resnet34': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        case 'resnet50': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        case 'resnet101':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        case 'resnet152': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        case 'vgg11': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        case 'vgg11_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        case 'vgg13': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
        case 'vgg13_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
        case 'vgg16': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        case 'vgg16_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
        case 'vgg19': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        case 'vgg19_bn': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
        case 'alexnet': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        case 'googlenet': 
            model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        case _: 
            raise ValueError(f'Unknown model: {name}')
    return model
