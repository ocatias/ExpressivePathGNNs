"""
Strongly based on https://github.com/gasmichel/PathNNs_expressive/blob/main/synthetic/main.py (MIT LICENSE)
"""

import random
import time 
import os

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from Exp.parser import parse_args
from Misc.config import config
from Misc.EXPdataset import EXPDataset
from Misc.SR25dataset import SR25Dataset
from Exp.preparation import get_transform, get_model

SR25_NAMES = [
    'sr16622',
    'sr251256',
    'sr261034',
    'sr281264',
    'sr291467',
    'sr361446',
    'sr401224'
]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def compute_failure_rates(args, data_loader, pws = True):
    device = args.device
    failure_rates = []

    print("Begin Evaluating.\n")
    for seed in [42, 69, 1337, 420, 666]:
        set_seed(seed)
        model = get_model(args, 1, 1, 1)
        model.to(device)
        embeddings = []
        model.eval()
        if pws:
            embds = [[],[]]
        else: 
            embds = []
        with torch.no_grad():
            for i, data in enumerate(tqdm((data_loader))):
                pre = model.compute_embedding(data.to(device))
                if pws:
                    embds[i%2].append(pre.detach().cpu())
                else:
                    embds.append(pre.detach().cpu())
                
        if pws:
            failure_rate = _isomorphism_pws(embds)
        else:
            failure_rate = _isomorphism(torch.cat(embds, 0).detach().cpu().numpy())
        print(f"Failure Rate: {failure_rate}")
        failure_rates.append(failure_rate)
    print("\n\nFINAL SUMMARY")
    mean = np.mean(failure_rates)
    std = np.std(failure_rates)
    print(f"{args.dataset}")
    print(f"Failure Rate: {mean:.4f} +/- {std:.4f}")
    return mean, std

def main(args):
    print(args)
    transform = get_transform(args)
    
    if args.dataset == "EXP":
        raw_file_name = os.path.join(config.DATA_PATH, "EXP", "raw", "EXP.txt")
        dir = os.path.join(config.DATA_PATH, args.dataset, repr(transform).replace("\n", ""))
        dataset = EXPDataset(root=dir, raw_file_name=raw_file_name, name=args.dataset, length=args.path_length, pre_transform=transform)  
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
        mean, std = compute_failure_rates(args, data_loader, pws=True)
        output_path = os.path.join(config.RESULTS_PATH, "exp.txt")
        with open(output_path, 'w') as file:
            file.write(f"EXP: {mean:.4f}; {std:.4f}")
        
    elif args.dataset == "SR25":
        for name in SR25_NAMES:
            raw_file_name = os.path.join(config.DATA_PATH, "SR25", "raw", f"{name}.g6")
            dir = os.path.join(config.DATA_PATH, args.dataset, name, repr(transform).replace("\n", ""))
            dataset = SR25Dataset(root=dir, raw_file_name=raw_file_name, name=name, length=args.path_length, pre_transform=transform)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            mean, std = compute_failure_rates(args, data_loader, pws=False)

            output_path = os.path.join(config.RESULTS_PATH, "sr25.txt")
            with open(output_path, 'w') as file:
                file.write(f"{name}: {mean:.4f}; {std:.4f}\n")

def _isomorphism_pws(preds, eps=1e-5, p=2):
    
    distinguishable = 0
    total = len(preds[0])
    for i in range(total):
        if torch.pdist(torch.concat((preds[0][i], preds[1][i]),dim=0), p=p) > eps:
            distinguishable += 1
            
    print(f" {distinguishable} / {total}")
    return float(total - distinguishable)/ total

def _isomorphism(preds, eps=1e-5, p=2):
    # NB: here we return the failure percentage... the smaller the better!
    assert preds is not None
    # assert preds.dtype == np.float64
    preds = torch.tensor(preds, dtype=torch.float64)
    mm = torch.pdist(preds, p=p)
    wrong = (mm < eps).sum().item()
    metric = wrong / mm.shape[0]
    return metric

def run(passed_args = None):
    args = parse_args(passed_args)
    assert args.batch_size == 1
    return main(args)

if __name__ == "__main__":
    run()
