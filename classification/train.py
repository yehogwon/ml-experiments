import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch

from common import create_model

def main(args: argparse.Namespace): 
    pass

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
    
    args = parser.parse_args()
    main(args)
