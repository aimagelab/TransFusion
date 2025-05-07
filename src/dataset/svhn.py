import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
import numpy as np

#import templates
from .templates import get_templates
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random


class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 few_shot=False):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')

        self.train_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        if few_shot:
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
            self.train_loader = DataLoader(self.train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers)

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.templates = get_templates('svhn')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'A photo of a {c}'