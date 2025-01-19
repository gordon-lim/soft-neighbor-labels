# Modified from: https://github.com/google-research/heldout-influence-estimation/blob/master/mnist-example/mnist_infl_mem.py
# Changes made:
# - Converted from JAX to PyTorch
# - Adapted to work with CIFAR-10
# - Held out only a subset of the data: all incorrect noisy labels and an equal number of sampled clean labels
# Place script in: cifar-10-100n-main

import os
import itertools

import numpy as np
import numpy.random as npr

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

# import wandb
import argparse
from models import *
from tqdm import tqdm
from data.datasets import input_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--seed_start', type=int, required=True, help='Starting seed')
parser.add_argument('--seed_end', type=int, required=True, help='Ending seed')
parser.add_argument('--gpu', type=int, help='GPU index to use', default=0)

global args
global npz_fn
# global wandb_pn
args = parser.parse_args()
npz_fn = f'results-infl-mem/cifar-{args.noise_type}-{"human" if args.is_human else "syn"}-{args.seed_start}-{args.seed_end}-infl-mem.npz'
# wandb_pn = f'cifar-{args.noise_type}-{"human" if args.is_human else "syn"}-infl-mem'

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(optimizer, epoch):
    alpha_plan = [0.1] * 40 + [0.01] * 40
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch] 

def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)

def batch_correctness(model, dataset, batch_size=64):

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    correct_preds = []
    label_preds = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct_preds.append(preds == targets)
            label_preds.append(preds)
    
    return torch.cat(correct_preds).cpu(), torch.cat(label_preds).cpu()

def load_cifar():

    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 
                      'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 
                      'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    noise_type = noise_type_map[args.noise_type]
    
    # load dataset
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')

    train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,noise_type,args.noise_path,args.is_human)
    print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])

    # Get noisy and clean idx
    train_noisy_idx = np.where(train_dataset.noise_or_not)[0]
    train_clean_idx = np.where(~train_dataset.noise_or_not)[0]
    
    # Always sample the same clean idx
    np.random.seed(0)
    sampled_clean_idx = np.random.choice(train_clean_idx, size=len(train_noisy_idx), replace=False)

    # Get rest idx that are not in train_noisy_idx and not in sampled_clean_idx
    all_idx = set(np.arange(len(train_dataset)))
    rest_idx = np.array(list(all_idx - set(train_noisy_idx) - set(sampled_clean_idx)))
    
    return dict(train_dataset=train_dataset, test_dataset=test_dataset,
                train_heldout_noisy_idx=train_noisy_idx,
                train_heldout_clean_idx=sampled_clean_idx,
                train_leavein_idx=rest_idx)

cifar_data = load_cifar()

def subset_train(seed, subset_ratio):
    torch.manual_seed(seed)
    
    # try:
    #     wandb.init(project=wandb_pn, config=args, name=f'seed_{seed}')
    # except Exception as e:
    #     print(f'W&B initialization failed: {e}')
    #     wandb.init(mode='disabled')  # This will disable W&B logging but keep the script running

    num_epochs = 80
    batch_size = 64

    # Sample random subset
    rng = npr.RandomState(seed)
    
    # Combine train_heldout_noisy_idx and train_heldout_clean_idx
    combined_heldout_idx = np.concatenate((cifar_data['train_heldout_noisy_idx'], 
                                           cifar_data['train_heldout_clean_idx']))
    
    # Sample 70% from the combined array
    sampled_combined_idx = rng.choice(combined_heldout_idx, 
                                      size=int(0.7 * len(combined_heldout_idx)), 
                                      replace=False)
    # Combine train_leavein_idx with the sampled idx
    subset_idx = np.concatenate((cifar_data['train_leavein_idx'], sampled_combined_idx))

    train_subset = Subset(cifar_data['train_dataset'], subset_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f'subset {seed} size:', len(train_subset))

    model = ResNet34(cifar_data['train_dataset'].nb_classes)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        n_batches = 0
        correct_preds = 0
        total_preds = 0

        adjust_learning_rate(optimizer, epoch)
        
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == targets).sum().item()
            total_preds += targets.size(0)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        # Log the loss and accuracy to wandb
        # wandb.log({'epoch': epoch, 'loss': running_loss/n_batches, 'train_accuracy': correct_preds/total_preds})

    # Save model
    model_save_path = f'weights/resnet34_subset_{seed}' + ('human' if args.is_human else 'syn') + '.pth'
    torch.save(model.state_dict(), model_save_path)
    # wandb.save(model_save_path)
    
    model.eval()
    with torch.no_grad():
            train_correct, train_preds = batch_correctness(model, cifar_data['train_dataset'])
            test_correct, test_preds = batch_correctness(model, cifar_data['test_dataset'])

    # Log final accuracy to wandb
    # wandb.log({'test_accuracy': test_correct.float().mean().item()})
    # wandb.finish()

    trainset_mask = np.zeros(len(cifar_data['train_dataset']), dtype=bool)
    trainset_mask[subset_idx] = True
    
    return trainset_mask, train_correct, train_preds, test_correct, test_preds

def estimate_infl_mem(seed_start, seed_end):
    
    subset_ratio = 0.7
    
    results = []
    for seed in tqdm(range(seed_start, seed_end + 1), desc=f'SS Ratio={subset_ratio:.2f}'):
        results.append(subset_train(seed, subset_ratio))

    trainset_mask = np.vstack([ret[0] for ret in results]) if len(results) > 1 else results[0][0]
    trainset_correctness = np.vstack([ret[1] for ret in results]) if len(results) > 1 else results[0][1]
    trainset_predictions = np.vstack([ret[2] for ret in results]) if len(results) > 1 else results[0][2]
    testset_correctness = np.vstack([ret[3] for ret in results]) if len(results) > 1 else results[0][3]
    testset_predictions = np.vstack([ret[4] for ret in results]) if len(results) > 1 else results[0][4]
        
    return dict(trainset_mask=trainset_mask, 
                trainset_correctness=trainset_correctness, 
                trainset_predictions=trainset_predictions,
                testset_correctness=testset_correctness,
                testset_predictions=testset_predictions)
  
def main():
    results = estimate_infl_mem(args.seed_start, args.seed_end)
    np.savez(npz_fn, **results)

if __name__ == '__main__':
    main()