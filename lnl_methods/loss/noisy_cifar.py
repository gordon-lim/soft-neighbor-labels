import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets
from typing import Optional, Callable, Tuple, Union, Any

class NoisyCIFAR10(datasets.CIFAR10):
    """
    Custom CIFAR10 class to add generated label noise
    """
    def __init__(
        self,
        train: bool,
        root: Union[str, Path] = './data',
        transform: Optional[Callable] = None,
        noisy_label_file: Optional[str] = None,
        seed: int = 100
    ):
        # Call the constructor of the base class
        super().__init__(
            root,
            train=train,
            transform=transform
        )

        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.given_labels = self.targets
        self.num_classes = len(self.classes)

        # Load filenames    
        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.filenames = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.filenames.extend(entry["filenames"])

        if noisy_label_file:
            print(f"Loading noisy labels from {noisy_label_file}...")
            noisy_label_df = pd.read_csv(noisy_label_file)
            self.noisy_labels = noisy_label_df['noisy_label'].tolist()
            assert self.targets == noisy_label_df['given_label'].tolist(), 'Misalignment error'
            self.targets = self.noisy_labels
        else:
            self.targets = self.given_labels.copy()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        target_label = self.targets[index]
        given_label = self.given_labels[index]
        return index, img, target_label, given_label

class NoisyCIFAR100(NoisyCIFAR10):
    """
    Source: https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py
    `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `NoisyCIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

# Acknowledgment: This work was assisted by using ChatGPT 4o, a language model developed by OpenAI.