import logging
try:
    import requests
except ImportError as e:
    logging.error("Please install requests using 'pip install requests'")
    raise e

import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from typing import Tuple

import yaml
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

#import templates
from .templates import get_templates
from pathlib import Path

class MyImagenetR(Dataset):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=False, transform=None,
                 target_transform=None, download=True) -> None:

        self.root = os.path.join(root, 'imagenet-r/')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.root):
            if download:
                # download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
                print("Downloading imagenet-r dataset...")
                url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
                r = requests.get(url, allow_redirects=True)
                if not os.path.exists(self.root):
                    os.makedirs(self.root)
                print("Writing tar on disk...")
                open(self.root + 'imagenet-r.tar', 'wb').write(r.content)
                print("Extracting tar...")
                os.system('tar -xf ' + self.root + 'imagenet-r.tar -C ' + self.root.rstrip('imagenet-r'))

                # move all files in imagenet-r to root with shutil
                import shutil
                print("Moving files...")
                for d in os.listdir(self.root + 'imagenet-r'):
                    shutil.move(self.root + 'imagenet-r/' + d, self.root)

                print("Cleaning up...")
                os.remove(self.root + 'imagenet-r.tar')
                os.rmdir(self.root + 'imagenet-r')

                print("Done!")
            else:
                raise RuntimeError('Dataset not found.')

        if self.train:
            data_config = yaml.load(open(f'{self.root}imagenet-r_train.yaml'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(f'{self.root}imagenet-r_test.yaml'), Loader=yaml.Loader)

        data = np.array(data_config['data'])
        self.data = []
        for i, item in enumerate(data):
            #replace 'data/imagenet-r' with self.root
            self.data.append(item.replace('data/imagenet-r/', self.root))
        self.targets = np.array(data_config['targets'])
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        return img, target
    

class IMAGENETR:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 few_shot=False):


        self.test_dataset = MyImagenetR(location, train=False, transform=preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.location = Path(location,'imagenet-r')
        self.class_names = self.get_class_names()

        self.templates = get_templates('cifar100')
        self.single_template = self.test_dataset.single_template = lambda c: f'A photo of a {c}'
       
    
    def get_class_names(self):
        with open(Path(self.location,'label_to_class_name.pkl'), 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = label_to_class_name.values()
        class_names = [x.replace('_', ' ') for x in class_names]
        self.class_names = class_names
        return self.class_names
