import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
import time
from utils.clip_wrapper import clip_img_wrap
import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import Custom_dataset
from utils.model_SimCLR import SimCLR_encoder
import torch.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
import utils.ResNet_for_32 as resnet_s
from utils.knn_utils import sample_soft_knn_labels
import argparse
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(diffusion_model, train_dataset, test_dataset, model_path, args, real_fp):  # Removed val_dataset
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    k = args.k
    # Added: Trust in given label for use in sample_soft_knn_labels
    alpha = args.alpha
    warmup_epochs = args.warmup_epochs

    embeddings_path = f'./embeddings/{args.fp_encoder}_{args.noise_type}.pt'
    
    if os.path.exists(embeddings_path):
        # If embeddings exist, load them
        print(f"Loading precomputed embeddings from {embeddings_path}")
        train_embed = torch.load(embeddings_path)
    else:
        # If embeddings don't exist, compute and save them
        print("Precomputing fp embeddings for training data")
        train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device, fp_dim=fp_dim).to(device)
        torch.save(train_embed, embeddings_path)
        print(f"Saved precomputed embeddings to {embeddings_path}")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Removed val_loader
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()
        train_loss = 0.0
        correct_cnt = 0
        all_cnt = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, data_indices] = data_batch[:3]

                if real_fp:
                    fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
                else:
                    fp_embd = train_embed[data_indices, :]

                # Added: Move following to correct device
                fp_embd = fp_embd.to(device)
                train_embed = train_embed.to(device)

                # Key difference
                soft_y_labels_batch = sample_soft_knn_labels(fp_embd, y_batch.to(device), train_embed,
                                                                  torch.tensor(train_dataset.targets).to(device),
                                                                  k=k, n_class=n_class, alpha=alpha)

                y_0_batch = soft_y_labels_batch.to(device)

                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
                n = x_batch.size(0)

                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)

                mse_loss = diffusion_loss(e, output)
                loss = torch.mean(mse_loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

                # Calculate training accuracy
                correct = cnt_agree(output.detach().cpu(), y_batch.cpu())#[0].item()
                correct_cnt += correct
                all_cnt += x_batch.shape[0]

        # Calculate average training loss and accuracy for this epoch
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_cnt / all_cnt

        # Test the model on the test set at the end of every epoch
        test_acc = test(diffusion_model, test_loader)  # Directly test on the test set after each epoch
        print(f"epoch: {epoch}, test accuracy: {test_acc:.2f}%")  # Print test accuracy each epoch

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_acc
        })
        
        # Save the model if test accuracy improves
        if test_acc > max_accuracy:
            print('Improved! Saving the model...')
            states = [diffusion_model.model.state_dict(),
                      diffusion_model.diffusion_encoder.state_dict(),
                      diffusion_model.fp_encoder.state_dict()]
            torch.save(states, model_path)
            print(f"Model saved at Epoch {epoch}, test accuracy: {test_acc}")
            max_accuracy = max(max_accuracy, test_acc)


def test(diffusion_model, test_loader):
    start = time.time()
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            [images, target, _] = data_batch[:3]
            target = target.to(device)

            label_t_0 = diffusion_model.reverse_ddim(images, stochastic=False, fq_x=None).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())#[0].item()
            correct_cnt += correct
            all_cnt += images.shape[0]

    print(f'time cost for CLR: {time.time() - start}')

    acc = 100 * correct_cnt / all_cnt
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', default='cifar10-pmd-0.35', help='noise label file', type=str)
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--device", default='cuda', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=100, help="k neighbors for knn", type=int)
    parser.add_argument("--alpha", default=0.3, help="trust in given label", type=float)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--fp_encoder", default='CLIP', help="which encoder for fp (SimCLR or CLIP)", type=str)
    parser.add_argument("--CLIP_type", default='ViT-L/14', help="which encoder for CLIP", type=str)
    parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    parser.add_argument("--seed", default=123, help="random seed", type=int)  # Added seed argument
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="lra+-diffusion", name=f'{args.noise_type}-seed{args.seed}', config=args)

    # Set the seed
    set_seed(args.seed)

    # set device
    device = args.device
    print('Using device:', device)

    dataset = args.noise_type.split('-')[0]

    # load datasets
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='./', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='./', train=False, download=True)

    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='./', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='./', train=False, download=True)
    else:
        raise Exception("Dataset should be cifar10 or cifar100")

    # load fp encoder
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        real_fp = True
        state_dict = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=torch.device(args.device))
        fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
        fp_encoder.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'CLIP':
        real_fp = False
        # fp_encoder = clip_img_wrap(args.CLIP_type, args.device, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

        clip_model_path = f'./model/CLIP_ViT_{dataset}.pt'
    
        if os.path.exists(clip_model_path):
            # Load CLIP model if it's saved
            fp_encoder = torch.load(clip_model_path, map_location=torch.device(args.device))
            print(f'Loaded CLIP model from {clip_model_path}')
        else:
            # Initialize CLIP model and save it for future use
            fp_encoder = clip_img_wrap(args.CLIP_type, args.device, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            torch.save(fp_encoder, clip_model_path)
            print(f'Saved CLIP model to {clip_model_path}')
            
        fp_dim = fp_encoder.dim
    else:
        raise Exception("fp_encoder should be SimCLR or CLIP")

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

    train_dataset = Custom_dataset(train_dataset_cifar.data, train_dataset_cifar.targets,  # Removed validation split
                                   transform=transform_train)
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets,
                                 transform=transform_test)

    batch_size = args.batch_size

    # load noisy label
    noise_label = np.load('../label_noise/' + args.noise_type + '.npy')
    print('Training on noise rate:', sum(train_dataset.targets != noise_label)/len(train_dataset))
    train_dataset.update_label(noise_label)
    print('Training on noise label:', args.noise_type)

    # initialize diffusion model
    model_path = f'./model/LRA+-diffusion_{args.fp_encoder}_{args.noise_type}.pt'
    diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
    
    diffusion_model.fp_encoder.eval()

    # train the diffusion model (Removed val_dataset as an argument)
    print(f'training LRA-diffusion using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'model saving dir: {model_path}')
    train(diffusion_model, train_dataset, test_dataset, model_path, args, real_fp=real_fp)  # Removed val_dataset