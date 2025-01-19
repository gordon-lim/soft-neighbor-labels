# LNL Methods 🛠️ 

We experimented with various methods for learning with noisy labels (LNL).

---

## Table of Contents
1. [LRA-Diffusion + SNLS](#1-lra-diffusion--snls)
2. [Progressive Label Correction (PLC)](#2-progressive-label-correction-plc)
3. [Co-teaching+](#3-co-teaching)
4. [Loss](#4-loss)

---

### 1. LRA-Diffusion + SNLS

**Folder**: `lra-diffusion-snls`

#### 1a. Vanilla LRA-Diffusion

We worked directly with the original LRA-Diffusion repository to evaluate their method on our CIFAR-10/CIFAR-100 noisy label sets.

**Steps to Reproduce Results**:  
Run the `train_CIFAR.py` script with the following command:

```bash
python train_CIFAR.py --noise_type cifar10-cluster-0.35 --seed ${seed} --device cuda:0
```
* noise_type: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

#### 1b. LRA-Diffusion + **SNLS (Ours)**

One of the key concepts in LRA-Diffusion is to sample a single, different label from neighboring images in the CLIP space as form of regularization. Building on this idea, we introduce Soft Neighbor Label Sampling, which generates a soft label distribution by leveraging the labels of the 100 nearest neighbors.

**Key change:**
The key change to use SNLS is in `utils/knn_utils.py`. We added the following function and used it in place of the original `sample_knn_labels`:
```python
def sample_soft_knn_labels(query_embd, y_query, prior_embd, labels, k=100, n_class=10, alpha=0.3):
    n_sample = query_embd.shape[0]
    
    # Get the top-k nearest neighbors
    distances, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # Get the labels of the 100 nearest neighbors
    neighbour_labels = labels[neighbour_ind]
    
    # One-hot encoding of neighbor labels
    y_one_hot_batch = nn.functional.one_hot(neighbour_labels, num_classes=n_class).float()

    # Count the frequency of each label across the neighbors
    label_frequencies = torch.sum(y_one_hot_batch, dim=1)

    # Normalize frequencies to get the soft label distribution
    label_frequencies /= torch.sum(label_frequencies, dim=1, keepdim=True)
    
    # Append the true label of the query to the distribution
    y_query_one_hot = nn.functional.one_hot(y_query, num_classes=n_class).float()
    soft_labels = (1 - alpha) * label_frequencies + alpha * y_query_one_hot

    return soft_labels
```

**Steps to Reproduce Results**:  
Run the `train_CIFAR_plus.py` script with the following command:

```bash
python train_CIFAR_plus.py --noise_type cifar10-cluster-0.35 --seed ${seed} --device cuda:0
```
* noise_type: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

---

### 2. Progressive Label Correction (PLC)

**Folder**: `plc`

We worked directly with the original PLC repository to evaluate their method on our CIFAR-10/CIFAR-100 noisy label sets.

**Steps to Reproduce Results**:  
Run the `train.py` script in `cifar/`:

```bash
python train.py --noisy_label_file ../../../label_noise/cifar10-pmd-0.70.npy --seed ${seed}
```
* noisy_label_file: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

---

### 3. Co-teaching+

**Folder**: `coteaching_plus`

We worked directly with the original `coteaching_plus` repository to evaluate their method on our CIFAR-10/CIFAR-100 noisy label sets.

**Steps to Reproduce Results**: 

Run the `main.py` script:

```bash
python main.py --dataset cifar10 --model_type coteaching_plus --noisy_label_file ../../label_noise/cifar10-cluster-0.70.csv --seed ${seed} --result_dir results/trial_${seed}/
```
* noisy_label_file: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

---

### 4. Loss

**Folder**: `loss`

#### 1a. GCELoss

**Steps to Reproduce Results**: 

Run the `trainer.py` script:

```bash
python trainer.py --noisy_label_file cifar10-cluster-0.70.csv --loss_function gce --seed ${seed} --gpu 1
```
* noisy_label_file: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

#### 1b. CrossEntropyLoss (Standard)

**Steps to Reproduce Results**: 

Run the `trainer.py` script:

```bash
python trainer.py --noisy_label_file cifar10-cluster-0.70.csv --loss_function cross_entropy --seed ${seed} --gpu 1
```
* noisy_label_file: Specify the noise configuration from ../../../label_noise/.
* seed: We used seed values 1, 2, and 3.

---
