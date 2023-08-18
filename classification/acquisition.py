import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

def _class_count(x: torch.Tensor, model: nn.Module, total: int) -> torch.Tensor: 
    probs = model(x)
    preds = probs.softmax(dim=1).argmax(dim=1)
    return torch.Tensor([(preds == i).sum() for i in range(total)])

def _class_count(dataset: VisionDataset, model: nn.Module, total: int, batch_size: int=128) -> torch.Tensor: 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = torch.zeros(total)
    for x, _ in loader: 
        counts += _class_count(x, model, total)
    return counts

def _class_balance_coefficient(counts: torch.Tensor) -> torch.Tensor: 
    total = counts.sum().item()
    return torch.exp(-counts / total)

def class_balance_sampling(dataset: VisionDataset, model: nn.Module, total: int, batch_size: int=64) -> list[float]: 
    counts = _class_count(dataset, model, total, batch_size)
    coeffs = _class_balance_coefficient(counts)
    weights = [coeffs[y] for _, y in dataset]
    return weights
