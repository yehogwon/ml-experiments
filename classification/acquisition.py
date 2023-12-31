import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

def _labels(dataset: VisionDataset, device: str, batch_size: int=32) -> torch.Tensor: 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    labels = None
    for _, y in tqdm(loader, desc='Extracting Labels'):
        y = y.to(device)
        if labels is None:
            labels = y
        else:
            labels = torch.cat((labels, y))
            del y
    return labels

def _class_count(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=128) -> torch.Tensor: 
    # device: the device on which the model is located
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    pred_stack = None
    with torch.no_grad():
        for x, _ in tqdm(loader, desc='Class Count'): 
            x = x.to(device)
            probs = F.softmax(model(x), dim=1).detach()
            # preds = torch.argmax(probs, dim=1) # It does not work for MPS (AMD GPUs)
            # The above exception was issued officially by PyTorch: https://github.com/pytorch/pytorch/issues/92311 and https://github.com/pytorch/pytorch/issues/98191
            preds = torch.max(probs, dim=1).indices # This works pretty well even for MPS
            if pred_stack is None:
                pred_stack = preds
            else:
                pred_stack = torch.cat((pred_stack, preds))
                del preds
            del x, probs
    return torch.Tensor([(pred_stack == i).sum().item() for i in range(total)]).to(device)

def class_balance_acquisition(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=32) -> torch.Tensor: 
    model.eval()
    counts = _class_count(dataset, model, total, device, batch_size) # (n_classes,)
    class_balances = counts / counts.sum() # (n_classes,)
    neg_exp_class_balances = torch.exp(-class_balances) # (n_classes,)
    weights = neg_exp_class_balances[_labels(dataset, device, batch_size).long()]
    del counts, class_balances, neg_exp_class_balances
    # return weights.tolist()
    # return list(enumerate(weights.tolist()))
    return weights

def bvsb_uncertainty(dataset: VisionDataset, model: nn.Module, total: int, device: str, batch_size: int=32) -> torch.Tensor: 
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    full_bvsb = None
    with torch.no_grad():
        for x, _ in tqdm(loader, desc='BvSB Uncertainty'): 
            x = x.to(device)
            probs = F.softmax(model(x), dim=1).detach()
            top_two = torch.topk(probs, k=2, dim=1).values.to(device)
            bvsb = top_two[:, 1] / top_two[:, 0]
            if full_bvsb is None:
                full_bvsb = bvsb
            else:
                full_bvsb = torch.cat((full_bvsb, bvsb))
                del bvsb
            del x, probs, top_two
    return full_bvsb
