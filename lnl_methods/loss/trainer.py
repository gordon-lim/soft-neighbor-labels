# -*- coding:utf-8 -*-
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from resnet import ResNet34
import argparse
import wandb
import noisy_cifar

# GCELoss implementation from https://github.com/hitcszx/ALFs/blob/master/lnl/losses.py
class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = torch.softmax(pred, dim=1)
        eps = 1e-6  # Small epsilon value to avoid numerical issues
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--noisy_label_file', type = str, default=None)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--gpu', type=int, help='GPU index to use', default=0)
parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'gce'], help='Choice of loss function: cross_entropy or gce')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choice of dataset: cifar10 or cifar100')

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]  

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = torch.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    memo_given_labeled, total_given_labeled = 0, 0
    memo_noisy_labeled, total_noisy_labeled = 0, 0

    for i, (sample_ids, images, target_labels, given_labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
       
        images = images.to(device)
        given_labels = given_labels.to(device)
        target_labels = target_labels.to(device)
       
        # compute outputs
        outputs = model(images)
        batch_loss = criterion(outputs, target_labels).mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        outputs = outputs.float()
        batch_loss = batch_loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, target_labels)[0]
        # get predicted label and max prob score
        probabilities = torch.softmax(outputs, dim=1)
        max_probs, predicted = torch.max(probabilities, dim=1)
        # update metrics
        losses.update(batch_loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        memo_indices = max_probs > 0.95
        target_match_given_indices = target_labels == given_labels
        
        memo_given_labeled += (memo_indices & target_match_given_indices).sum().item()
        memo_noisy_labeled += (memo_indices & ~target_match_given_indices).sum().item()
    
        total_given_labeled += target_match_given_indices.sum().item()
        total_noisy_labeled += (~target_match_given_indices).sum().item()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

        torch.cuda.empty_cache()

    memo_given_labeled_frac = memo_given_labeled / total_given_labeled if total_given_labeled > 0 else 0
    memo_noisy_labeled_frac = memo_noisy_labeled / total_noisy_labeled if total_noisy_labeled > 0 else 0
    
    return top1.avg, losses.avg, memo_given_labeled_frac, memo_noisy_labeled_frac

# Evaluate the Model
def evaluate(testloader, model, criterion, epoch=None, is_last_epoch=True):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    predictions = []
    with torch.no_grad():
        for i, (sample_ids, images, target_labels, given_labels) in enumerate(testloader):
            images = images.to(device)
            given_labels = given_labels.to(device)
            target_labels = target_labels.to(device)

            # compute outputs
            outputs = model(images)
            logits = outputs.data
            sample_loss = criterion(outputs, target_labels)
            batch_loss = sample_loss.mean()

            outputs = outputs.float()
            batch_loss = batch_loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, target_labels)[0]
            losses.update(batch_loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(testloader), batch_time=batch_time, loss=losses,
                          top1=top1))
                
            if is_last_epoch:
                for i, sample_id in enumerate(sample_ids):
                    predictions.append({
                    'sample_id': sample_id.item(),
                    'target_label': target_labels[i].item(),  
                    'given_label': given_labels[i].item(),
                    'loss': sample_loss[i].item(),
                    'logits': logits[i].tolist()
                })

            torch.cuda.empty_cache()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    try:
        wandb.log({'test accuracy': top1.avg, 'test loss': losses.avg}, step=epoch)
    except BrokenPipeError as e:
        print(f"Warning: Failed to log to wandb due to a BrokenPipeError: {e}")

    if is_last_epoch:
        if args.noisy_label_file:
            json_file_path = 'outputs/'+args.noisy_label_file.replace('.csv', '_reproduce_preds.json')
        else:
            json_file_path = 'outputs/cifar10_train_reproduce_preds.json'
        with open(json_file_path, 'w') as f:
            json.dump(predictions, f, indent=4)

    return top1.avg, losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(outputs, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#####################################main code ################################################
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

run_name = args.noisy_label_file.replace('.csv', '') + '-seed' + str(args.seed)
run = wandb.init(project=args.loss_function, name=run_name, config=args)

# Hyper Parameters
learning_rate = args.lr

if args.dataset == 'cifar10':
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    noisy_dataset_class = noisy_cifar.NoisyCIFAR10
elif args.dataset == 'cifar100':
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    noisy_dataset_class = noisy_cifar.NoisyCIFAR100

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

noisy_label_file = '../label_noise/' + args.noisy_label_file if args.noisy_label_file else None
    
trainset = noisy_dataset_class(root='../data',
                               train=True,
                               transform=train_transform,
                               noisy_label_file=noisy_label_file)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.workers)

testset = noisy_dataset_class(root='../data',
                              train=False,
                              transform=test_transform,
                              noisy_label_file=None)

testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.workers)

# load model
print('building model...')
model = ResNet34(trainset.num_classes)
print('building model done')
if args.loss_function == 'cross_entropy':
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
elif args.loss_function == 'gce':
    criterion = GCELoss(num_classes=trainset.num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

alpha_plan = [0.1] * 50 + [0.01] * 50 + [0.001] * 100
model = model.to(device)

model_stats = {
    'epoch': 0,
    'test_accuracy': 0,
    'train_accuracy': 0,
    'test_loss': 0,
    'train_loss': 0,
    'state_dict': None
}

# training
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc, train_loss, memo_given_labeled_frac, memo_noisy_labeled_frac = train(epoch, trainloader, model, criterion, optimizer)
    # evaluate models
    is_last_epoch = (epoch == args.n_epoch)
    test_acc, test_loss = evaluate(testloader, model, criterion, epoch, is_last_epoch)
    # log to wandb
    try:
        wandb.log({'train accuracy': train_acc, 'train loss': train_loss,
               'test accuracy': test_acc, 'test loss': test_loss,
               'memorized given labels': memo_given_labeled_frac,
               'memorized noisy labels': memo_noisy_labeled_frac}, step=epoch)
    except BrokenPipeError as e:
        print(f"Warning: Failed to log to wandb due to a BrokenPipeError: {e}")

    # Save best model weights and stats
    if test_acc > model_stats['test_accuracy']:
        model_stats.update({
            'epoch': epoch,
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'train_loss': train_loss,
            'state_dict': model.state_dict()
        })

        # Save the best stats
        torch.save(model_stats, 'best_model_and_stats.pth')