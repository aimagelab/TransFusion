import os
import torch
import torchvision.datasets as datasets
from collections import defaultdict
import random
from torch.utils.data import DataLoader, Subset

#import templates
from .templates import get_templates

class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 few_shot=False):
        # Data loading code
        traindir = os.path.join(location, 'dtd', 'train')
        valdir = os.path.join(location, 'dtd', 'val')
        
        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
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
        
        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.class_names = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        
        self.templates = get_templates('dtd')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'A photo of a {c}'