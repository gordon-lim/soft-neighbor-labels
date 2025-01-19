import sys
sys.path.append('..')

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

import random
import os
import copy
import argparse
from tqdm import tqdm
import numpy as np
from cifar_train_val_test import CIFAR10, CIFAR100
from termcolor import cprint

from utils import lrt_correction
from networks.preact_resnet import preact_resnet34
from noise import noisify_with_P, noisify_cifar10_asymmetric, noisify_cifar100_asymmetric
import wandb  # WandB integration

# Random seed setting
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)

def main(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # Ensure determinism for reproducibility

    noise_label_path = args.noisy_label_file
    noise_y = np.load(noise_label_path)
    print(f'Load noisy label from {noise_label_path}')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    batch_size = 128
    num_workers = 2
    lr = args.lr
    current_delta = args.delta

    which_data_set = os.path.basename(args.noisy_label_file).split('-')[0]
    noise_level = args.noise_level
    noise_type = args.noise_type

    # Load CIFAR10/100 dataset with original class
    if which_data_set == 'cifar10':

        # Data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = CIFAR10(root='data', split='train', train_ratio=1.0, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        testset = CIFAR10(root='data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 10
        in_channel = 3
    elif which_data_set == 'cifar100':

        # Data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        trainset = CIFAR100(root='data', split='train', train_ratio=1.0, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        testset = CIFAR100(root='data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 100
        in_channel = 3
    else:
        raise ValueError('Dataset should be cifar10 or cifar100.')

    print(f'train data size: {len(trainset)}')
    print(f'test data size: {len(testset)}')

    # Sanity check
    num_noise_class = len(np.unique(noise_y))
    assert num_noise_class == num_class, "The number of noisy labels does not match the dataset's class count!"
    assert len(noise_y) == len(trainset), "The number of noisy labels does not match the training data size!"

    # Generate noise
    gt_clean_y = copy.deepcopy(trainset.get_data_labels())
    y_train = noise_y.copy()

    noise_y_train, p = None, None

    if noise_type == "uniform":
        noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level,
                                                        random_state=random_seed)
        trainset.update_corrupted_label(noise_y_train)
        print("Applied uniform noise")
    else:
        if which_data_set == 'cifar10':
            noise_y_train, p, _ = noisify_cifar10_asymmetric(y_train, noise=noise_level, random_state=random_seed)
        elif which_data_set == 'cifar100':
            noise_y_train, p, _ = noisify_cifar100_asymmetric(y_train, noise=noise_level, random_state=random_seed)
        trainset.update_corrupted_label(noise_y_train)
        print("Applied asymmetric noise")
    
    print(f"Probability transition matrix:\n{p}")

    real_noise_level = np.sum(noise_y_train != gt_clean_y) / len(noise_y_train)
    print(f'\n>> Real Noise Level: {real_noise_level}')
    y_train_tilde = copy.deepcopy(noise_y_train)
    y_syn = copy.deepcopy(noise_y_train)

    # WandB initialization
    wandb.init(project="plc", config=args)

    # Set up network, optimizer, and scheduler
    f = preact_resnet34(num_input_channels=in_channel, num_classes=num_class)
    f = f.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)

    f_record = torch.zeros([args.rollWindow, len(y_train_tilde), num_class])

    best_acc = 0
    best_epoch = 0
    best_weights = None

    # -- Training Loop --
    for epoch in range(args.nepoch):
        train_loss = 0
        train_correct = 0
        train_total = 0

        f.train()
        for _, (features, labels, _, indices) in enumerate(tqdm(trainloader, ascii=True, ncols=50)):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = f(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

        train_acc = train_correct / train_total * 100
        cprint(f"Epoch [{epoch+1}|{args.nepoch}] \t Train Acc {train_acc:.3f}%", "yellow")

        # -- Evaluate on Test Dataset --
        f.eval()
        test_total = 0
        test_correct = 0

        with torch.no_grad():
            for images, labels, _, _ in testloader:
                images, labels = images.to(device), labels.to(device)

                outputs = f(images)
                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = test_correct / test_total * 100.
        cprint(f'>> Epoch [{epoch+1}|{args.nepoch}] \t Test accuracy: {test_acc:.3f}%', 'cyan')

        # Log test metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
        })

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets
            y_corrected, current_delta = lrt_correction(np.array(y_tilde).copy(), f_x, current_delta=current_delta, delta_increment=args.inc)

            trainset.update_corrupted_label(y_corrected)

        scheduler.step()

    # -- Final Testing --
    cprint('>> Testing with the best validation model <<', 'cyan')
    test_total = 0
    test_correct = 0

    f.eval()
    with torch.no_grad():
        for images, labels, _, _ in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = f(images)

            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.
    print(f'>> Final test accuracy: {test_acc:.3f}%', 'cyan')
    wandb.log({"final_test_accuracy": test_acc})

    print(f"\nFinal Clean Ratio: {sum(np.array(trainset.targets).flatten() == np.array(y_syn).flatten()) / float(len(np.array(y_syn))) * 100:.3f}%")
    print(f"Final Test Accuracy: {test_acc:.3f}%")
    print(f"Final Delta Used: {current_delta}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_label_file', type=str, default=None)
    parser.add_argument('--noise_type', default='uniform', help='noise type [uniform | asym]', type=str)
    parser.add_argument('--noise_level', default=0.0, help='noise level [for additional uniform/asymmetric noise applied to the PMD noise]', type=float)
    parser.add_argument("--delta", default=0.3, help="initial threshold", type=float)
    parser.add_argument("--nepoch", default=180, help="number of training epochs", type=int)
    parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--gpus", default=0, help="which GPU to use", type=int)
    parser.add_argument("--warm_up", default=8, help="warm-up period", type=int)
    parser.add_argument("--seed", default=77, help="random seed", type=int)
    parser.add_argument("--lr", default=0.01, help="initial learning rate", type=float)
    parser.add_argument("--inc", default=0.1, help="increment", type=float)
    args = parser.parse_args()
    main(args)