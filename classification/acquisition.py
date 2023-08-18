import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

def _class_count(x: torch.Tensor, model: nn.Module, total: int) -> torch.Tensor: 
    probs = model(x)
    preds = F.softmax(probs, dim=1).argmax(dim=1)
    return torch.Tensor([(preds == i).sum() for i in range(total)]) # FIXME: this loop is a bottleneck

def _class_count_in_dataset(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=128) -> torch.Tensor: 
    # device: the device on which the model is located
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = torch.zeros(total)
    for x, _ in tqdm(loader, desc='Computing the class balance in the dataset according to the model prediction'): 
        x = x.to(device)
        counts += _class_count(x, model, total)
    return counts

def _class_balance_coefficient(counts: torch.Tensor) -> torch.Tensor: 
    total = counts.sum().item()
    return torch.exp(-counts / total)

def class_balance_sampling(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=64) -> list[tuple[int, float]]: 
    counts = _class_count_in_dataset(dataset, model, total, device, batch_size)
    coeffs = _class_balance_coefficient(counts)
    weights = [coeffs[y] for _, y in dataset]
    return list(enumerate(weights))
