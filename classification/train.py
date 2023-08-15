import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision import transforms

import wandb

from dataset.common import n_classes, dataset
from model import Classifier

from tqdm import tqdm

class Trainer: 
    def __init__(self, exp_name: str, dataset: VisionDataset, model: nn.Module, ckpt_path: str, ckpt_interval: int, device: str='cpu') -> None:
        self.exp_name = exp_name
        self.dataset = dataset
        self.model = model
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval
        self.device = device

        wandb.init(project='Classification Experiment', name=self.exp_name)

    def train(self, batch_size: int, n_epoch: int, lr: float, weight_decay: float, start_epoch: int) -> None: 
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(start_epoch, n_epoch + 1):
            losses = []
            for _, (x, y) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}/{n_epoch}', total=len(train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                wandb.log({'loss': loss.item()})

            loss_avg = sum(losses) / len(losses)
            print(f'Epoch {epoch} | Loss: {loss_avg}')
            wandb.log({'epoch': epoch, 'loss_avg': loss_avg})
            
            if epoch % self.ckpt_interval == 0:
                ckpt_saved = os.path.join(self.ckpt_path, f'{self.exp_name}_{epoch}.pth')
                torch.save(self.model.state_dict(), ckpt_saved)
                print(f'Checkpoint saved: {ckpt_saved}')

class ActiveLearningTrainer(Trainer):
    def __init__(self, exp_name: str, dataset: VisionDataset, model: torch.nn.Module, ckpt_path: str, ckpt_interval: int, n_stages: int, budget_per_stage: int, cost_function: str, device: str='cpu') -> None:
        super().__init__(exp_name, dataset, model, ckpt_path, ckpt_interval, device)
        self.n_stages = n_stages
        self.budget_per_stage = budget_per_stage

        self.acquisition_function = None # TODO: acquisition function

        # TODO: cost function

    def train(self) -> None: 
        # TODO: implement training pipeline for active learning for image classification
        pass

    def pick_k(self, k: int) -> list[int]: 
        pass

def main(args: argparse.Namespace): 
    model = Classifier(args.model, n_classes(args.dataset))

    if args.pretrained_model:
        print(f'Load pretrained model: {args.pretrained_model}')
        model.load_state_dict(torch.load(args.pretrained_model))
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_ = dataset(args.dataset, train=True, transform=transform)

    if args.al:
        al = ActiveLearningTrainer(args.exp_name, dataset_, model, args.ckpt_path, args.ckpt_interval, args.al_stage, args.budget_per_stage, args.cost_function, device=args.device)
        al.train(args.batch_size, args.epochs, args.lr, args.weight_decay)
    else:
        trainer = Trainer(args.exp_name, dataset_, model, args.ckpt_path, args.ckpt_interval, device=args.device)
        trainer.train(args.batch_size, args.epochs, args.lr, args.weight_decay, args.start_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--ckpt_interval', type=int, default=10)
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--start_epoch', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--al', action='store_true')
    parser.add_argument('--al_stage', type=int)
    parser.add_argument('--budget_per_stage', type=int)
    parser.add_argument('--cost_function', type=str)

    parser.add_argument('--exp_name', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
