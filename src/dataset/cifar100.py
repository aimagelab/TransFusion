import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from torch.utils.data import DataLoader, Subset

from collections import defaultdict
import random

class CIFAR100:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=2, few_shot = False):
        if few_shot:
            self.train_dataset = PyTorchCIFAR100(root=location, download=True, train=True, transform=preprocess)
            # Sample few examples per class for training
            class_indices = defaultdict(list)
            for idx, (_, label) in enumerate(self.train_dataset):
                class_indices[label].append(idx)

            # Limit to 'samples_per_class' per class
            sampled_indices = []
            for indices in class_indices.values():
                sampled_indices.extend(random.sample(indices, min(10, len(indices))))

            self.train_dataset_subset = Subset(self.train_dataset, sampled_indices)
            self.train_loader = DataLoader(self.train_dataset_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            self.train_dataset = PyTorchCIFAR100(root=location, download=True, train=True, transform=preprocess)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = PyTorchCIFAR100(root=location, download=True, train=False, transform=preprocess)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.class_names = self.test_dataset.classes  # Same as train_dataset.classes
       
        self.templates = self.test_dataset.templates = self.train_dataset.templates = [
            lambda c:f'a photo of a {c}.',
            lambda c:f'a blurry photo of a {c}.',
            lambda c:f'a black and white photo of a {c}.',
            lambda c:f'a low contrast photo of a {c}.',
            lambda c:f'a high contrast photo of a {c}.',
            lambda c:f'a bad photo of a {c}.',
            lambda c:f'a good photo of a {c}.',
            lambda c:f'a photo of a small {c}.',
            lambda c:f'a photo of a big {c}.',
            lambda c:f'a photo of the {c}.',
            lambda c:f'a blurry photo of the {c}.',
            lambda c:f'a black and white photo of the {c}.',
            lambda c:f'a low contrast photo of the {c}.',
            lambda c:f'a high contrast photo of the {c}.',
            lambda c:f'a bad photo of the {c}.',
            lambda c:f'a good photo of the {c}.',
            lambda c:f'a photo of the small {c}.',
            lambda c:f'a photo of the big {c}.',
        ]
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'A photo of a {c}'
