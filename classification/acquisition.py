import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

def _class_count(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=128) -> torch.Tensor: 
    # device: the device on which the model is located
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_stack = None
    for x, _ in loader: 
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        # preds = torch.argmax(probs, dim=1) # It does not work for MPS (AMD GPUs)
        # The above exception was issued officially by PyTorch: https://github.com/pytorch/pytorch/issues/92311 and https://github.com/pytorch/pytorch/issues/98191
        preds = torch.max(probs, dim=1).indices # This works pretty well even for MPS
        if pred_stack is None:
            pred_stack = preds
        else:
            pred_stack = torch.cat((pred_stack, preds))
    return torch.Tensor([(pred_stack == i).sum().item() for i in range(total)])

def class_balance_acquisition(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=32) -> list[tuple[int, float]]: 
    counts = _class_count(dataset, model, total, device, batch_size) # (n_classes,)
    class_balances = counts / counts.sum() # (n_classes,)
    weights = class_balances[torch.Tensor([y for _, y in dataset]).long()].tolist()
    return list(enumerate(weights))
