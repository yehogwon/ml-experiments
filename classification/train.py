import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

import wandb

from common import create_model

class Trainer: 
    def __init__(self, exp_name: str, dataset: VisionDataset, model: nn.Module, ckpt_path: str, device: str='cpu') -> None:
        self.exp_name = exp_name
        self.dataset = dataset
        self.model = model
        self.ckpt_path = ckpt_path
        self.device = device

        wandb.init(project='Classification Experiment', name=self.exp_name)

    def train(self, batch_size: int, n_epoch: int, lr: float, weight_decay: float) -> None: 
        # TODO: implement training pipeline for image classification
        pass

class ActiveLearningTrainer(Trainer):
    def __init__(self, exp_name: str, dataset: VisionDataset, model: torch.nn.Module, ckpt_path: str, n_stages: int, budget_per_stage: int, cost_function: str, device: str='cpu') -> None:
        super().__init__(self, exp_name, dataset, model, ckpt_path, device)
        self.n_stages = n_stages
        self.budget_per_stage = budget_per_stage

        # TODO: cost function

    def train(self) -> None: 
        # TODO: implement training pipeline for active learning for image classification
        pass

def main(args: argparse.Namespace): 
    model = create_model(args.model)

    if args.pretrained_model:
        print(f'Load pretrained model: {args.pretrained_model}')
        model.load_state_dict(torch.load(args.pretrained_model))
    
    if args.al:
        al = ActiveLearningTrainer(args.exp_name, args.dataset, model, args.ckpt_path, args.al_stage, args.budget_per_stage, args.cost_function)
        al.train(args.batch_size, args.epochs, args.lr, args.weight_decay)
    else:
        trainer = Trainer(args.exp_name, args.dataset, model, args.ckpt_path)
        trainer.train(args.batch_size, args.epochs, args.lr, args.weight_decay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--pretrained_model', type=str)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--al', action='store_true')
    parser.add_argument('--al_stage', type=int)
    parser.add_argument('--budget_per_stage', type=int)
    parser.add_argument('--cost_function', type=str)

    parser.add_argument('--exp_name', type=str)
    
    args = parser.parse_args()
    main(args)
