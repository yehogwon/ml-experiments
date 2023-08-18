import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision import transforms

import wandb

from dataset.common import n_classes, create_dataset
from model import Classifier
from acquisition import *

from tqdm import tqdm

class Trainer: 
    def __init__(self, exp_name: str, dataset: str, transform, model: nn.Module, ckpt_path: str, ckpt_interval: int, device: str='cpu') -> None:
        self.exp_name = exp_name
        self.dataset_name = dataset
        self.transform = transform
        self.train_dataset = create_dataset(dataset, train=True, transform=transform)
        self.test_dataset = create_dataset(dataset, train=False, transform=transform)

        self.model = model
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval
        self.device = device

    def train(self, batch_size: int, n_epoch: int, lr: float, weight_decay: float, start_epoch: int) -> None: 
        wandb.init(project='Classification Experiment', name=self.exp_name, config={
            'dataset': self.dataset_name,
            'model': self.model.__class__.__name__,
            'batch_size': batch_size,
            'n_epoch': n_epoch,
            'lr': lr,
            'weight_decay': weight_decay,
            'start_epoch': start_epoch,
            'device': self.device,
            'ckpt_path': self.ckpt_path,
            'ckpt_interval': self.ckpt_interval
        })

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(start_epoch, n_epoch + 1):
            train_loss, train_acc = self._train_iteration(train_loader, loss_fn, optimizer, desc=f'Epoch {epoch}/{n_epoch}', wandb_log=True)

            log_info = {
                'epoch': epoch, 
                'train_loss': train_loss, 
                'train_acc': train_acc
            }

            val_acc, val_loss = self.validate(batch_size) # run validation only on CIFAR-10/100
            log_info.update({
                'val_loss': val_loss, 
                'val_acc': val_acc
            })

            print(' | '.join([f'{k.title()}: {v}' for k, v in log_info.items()]))
            wandb.log(log_info)

            if epoch % self.ckpt_interval == 0:
                ckpt_path = self._save_model(f'{self.exp_name}_{epoch}.pth')
                print(f'Checkpoint saved: {ckpt_path}')
    
    def _train_iteration(self, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, desc: str='Training', wandb_log: bool=True) -> tuple[float, float]:
        losses = []
        n_correct = 0
        
        for x, y in tqdm(dataloader, desc=desc):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(x)
            n_correct += (F.softmax(y_pred, dim=1).argmax(dim=1) == y).sum().item()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if wandb_log: 
                wandb.log({'loss': loss.item()})
        
        return sum(losses) / len(losses), n_correct / len(self.train_dataset) # loss_avg, acc
    
    def validate(self, batch_size: int) -> float: 
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        loss_fn = nn.CrossEntropyLoss()
        n_correct = 0
        losses = []
        for x, y in tqdm(test_loader, desc=f'Validation'):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            n_correct += (F.softmax(y_pred, dim=1).argmax(dim=1) == y).sum().item()

            loss = loss_fn(y_pred, y)
            losses.append(loss.item())

        acc = n_correct / len(self.test_dataset)
        loss_avg = sum(losses) / len(losses)
        return acc, loss_avg

    def _save_model(self, name: str) -> str: 
        ckpt_saved = os.path.join(self.ckpt_path, name)
        torch.save(self.model.state_dict(), ckpt_saved)
        return ckpt_saved # return the path to saved checkpoint

class ActiveLearningTrainer(Trainer):
    def __init__(self, exp_name: str, dataset: str, transform, model: nn.Module, ckpt_path: str, ckpt_interval: int, acquisition_function: str, device: str='cpu') -> None:
        super().__init__(exp_name, dataset, transform, model, ckpt_path, ckpt_interval, device)

        match acquisition_function:
            case 'class_balance_sampling':
                self.acquisition_function = class_balance_sampling
            case _: 
                raise ValueError(f'Invalid acquisition function: {acquisition_function}')
        # self.acquisition_function: (dataset: VisionDataset, model: nn.Module, total: int, **kwargs) -> list[tuple[int, float]]

        self.acquisition_values = [0] * len(self.train_dataset) # list of tuples (index, acquisition value) <- acquisition values of labeled samples are set to 0
        self.labeled_ones = np.zeros(len(self.train_dataset), dtype=int) # 0: unlabeled, 1: labeled e.g., self.labeled_ones[i] = 1 iff i-th sample is labeled
        # np.where(self.labeled_ones == 1)[0].tolist() returns the list of indices of labeled samples
        # np.count_nonzero(self.labeled_ones) returns the number of labeled samples

    def train(self, batch_size: int, n_epoch: int, lr: float, weight_decay: float, start_epoch: int, n_stages: int, budget_per_stage: int) -> None: 
        wandb.init(project='Classification Experiment', name=self.exp_name, config={
            'dataset': self.dataset_name,
            'model': self.model.__class__.__name__,
            'batch_size': batch_size,
            'n_epoch': n_epoch,
            'lr': lr,
            'weight_decay': weight_decay,
            'start_epoch': start_epoch,
            'device': self.device,
            'n_stages': n_stages,
            'budget_per_stage': budget_per_stage,
            'ckpt_path': self.ckpt_path,
            'ckpt_interval': self.ckpt_interval
        })

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.model.to(self.device)
        self.model.train()

        for stage in range(1, n_stages + 1): 
            # sampling: get indices to train on
            # label_indices: np.ndarray <- indices to newly label
            if stage == 1: 
                assert self._remaining_samples() >= budget_per_stage, f'Not enough samples to label: {self._remaining_samples()} < {budget_per_stage}'
                label_indices = np.random.choice(np.where(self.labeled_ones == 0)[0], size=budget_per_stage, replace=False)
            else: 
                if self._remaining_samples() < budget_per_stage: 
                    label_indices = np.where(self.labeled_ones == 0)[0]
                else: 
                    label_indices = self._update_acquisition()[:budget_per_stage]
            # For the reference of SubsetRandomSampler, refer to here: https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, sampler=torch.utils.data.SubsetRandomSampler(label_indices))

            self.model.to(self.device)
            self.model.train()

            train_losses, train_acces = [], []
            val_losses, val_acces = [], []
            for epoch in range(start_epoch, n_epoch + 1):
                train_loss, train_acc = self._train_iteration(train_loader, loss_fn, optimizer, desc=f'Epoch {epoch}/{n_epoch}', wandb_log=True)

                log_info = {
                    f'stage{stage}/epoch': epoch, 
                    f'stage{stage}/train_loss': train_loss, 
                    f'stage{stage}/train_acc': train_acc
                }

                val_acc, val_loss = self.validate(batch_size) # run validation only on CIFAR-10/100
                log_info.update({
                    f'stage{stage}/val_loss': val_loss, 
                    f'stage{stage}/val_acc': val_acc
                })

                print(' | '.join([f'{k.title()}: {v}' for k, v in log_info.items()]))
                wandb.log(log_info)

                train_losses.append(train_loss)
                train_acces.append(train_acc)
                val_losses.append(val_loss)
                val_acces.append(val_acc)

                if epoch % self.ckpt_interval == 0:
                    ckpt_path = self._save_model(f'{self.exp_name}_{epoch}.pth')
                    print(f'Checkpoint saved: {ckpt_path}')

            wandb.log({
                'stage': stage,
                f'train_loss_avg': sum(train_losses) / len(train_losses),
                f'train_acc_avg': sum(train_acces) / len(train_acces),
                f'val_loss_avg': sum(val_losses) / len(val_losses),
                f'val_acc_avg': sum(val_acces) / len(val_acces)
            })
    
    def _update_acquisition(self) -> list[int]: 
        acquisition_value_arr = np.array(self.acquisition_function(self.train_dataset, self.model, n_classes(self.dataset_name)), dtype=float)
        # masked_acquisition_values = (cal_acquisition_value * (1 - self.labeled_ones)).tolist()
        acquisition_value_arr.T[1][np.where(self.labeled_ones == 1)[0]] = 0 # masked acquisition values (labeled -> 0)
        sorted_masked_acquisition_arr = acquisition_value_arr[acquisition_value_arr[:, 1].argsort()[::-1]] # sort by acquisition values (descending order)
        sorted_indices = sorted_masked_acquisition_arr[:, 0].astype(int).tolist()
        sorted_values = sorted_masked_acquisition_arr[:, 1].tolist()
        self.acquisition_values = list(zip(sorted_indices, sorted_values))
        return sorted_indices
    
    def _fully_labeled(self) -> bool: 
        return np.count_nonzero(self.labeled_ones) == len(self.train_dataset)
    
    def _remaining_samples(self) -> int: 
        return len(self.train_dataset) - np.count_nonzero(self.labeled_ones)

def main(args: argparse.Namespace): 
    model = Classifier(args.model, n_classes(args.dataset), pretrained=not args.scratch_backbone)

    if args.pretrained_model:
        print(f'Load pretrained model: {args.pretrained_model}')
        model.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    if args.al:
        trainer = ActiveLearningTrainer(args.exp_name, args.dataset, transform, model, args.ckpt_path, args.ckpt_interval, args.acquisition_function, device=args.device)
    else:
        trainer = Trainer(args.exp_name, args.dataset, transform, model, args.ckpt_path, args.ckpt_interval, device=args.device)

    if args.validate: 
        acc = trainer.validate(args.batch_size)
        print(f'Accuracy: {acc}')
    else: 
        if args.al: 
            trainer.train(args.batch_size, args.epochs, args.lr, args.weight_decay, args.start_epoch, args.al_stage, args.budget_per_stage)
        else: 
            trainer.train(args.batch_size, args.epochs, args.lr, args.weight_decay, args.start_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name to train/validate')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to save checkpoints (directory)')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='interval for saving checkpoints')
    parser.add_argument('--model', type=str, default='resnet34', help='model architecture (backbone)')
    parser.add_argument('--device', type=str, default='cpu', help='device on which the model will be trained/validated')

    parser.add_argument('--scratch_backbone', action='store_true', help='whether to train the backbone from scratch')
    parser.add_argument('--pretrained_model', type=str, help='path to pretrained model')
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch for training')
    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization coefficient')

    parser.add_argument('--validate', action='store_true', help='whether to validate the model (pretrained model required)')
    
    parser.add_argument('--al', action='store_true', help='whether to adopt active learning framework')
    parser.add_argument('--acquisition_function', type=str, help='acquisition function for selecting samples to label', default='class_balance_sampling')
    parser.add_argument('--al_stage', type=int, help='number of stages for active learning')
    parser.add_argument('--budget_per_stage', type=int, help='total cost of labeling samples per stage')

    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for wandb and ckpt')
    
    args = parser.parse_args()
    main(args)
