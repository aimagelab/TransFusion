"""
Dataset classes and utilities for loading and preprocessing datasets.
Supports various vision datasets and custom dataset logic.
"""
######################################################################
# Dataset Classes and Utilities
######################################################################

import io
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
import requests
import torch

import clip
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100


try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except ImportError:
    raise ImportError(
        "Please install the google_drive_downloader package by running: `pip install googledrivedownloader`")


def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(
        os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files


class utk_face(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, label="gender", attribute='race', thegroup=None, subset='Training'):
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.subset = subset
        self.race_mapping = {0: "White", 1: "Black",
                             2: "Asian", 3: "Indian", 4: "Others"}
        self.gender_mapping = {0: "Male", 1: "Female"}
        self.label = label.lower()
        self.attribute = attribute.lower()

        if files is not None:
            self.files = files
        else:
            self.files = sorted(find_all_files(self.dataset_dir, suffix='jpg'))

        if self.attribute and thegroup:
            attribute_mapping = {'age': 0, 'gender': 1, 'race': 2}
            tmp_files = []
            for file in self.files:
                labels = file.split("_")
                group = int(labels(attribute_mapping[self.attribute]))
                if group == thegroup:
                    tmp_files.append(file)
            self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = Path(self.dataset_dir, self.files[idx])
        image = self.preprocess(Image.open(file_path))
        labels = self.files[idx].split("_")
        age = int(labels[0])
        gender = int(labels[1])
        race = int(labels[2])
        if self.label == 'age':
            token = f"A photo of a {age} years old person"
            token = clip.tokenize(token)
            token = token.squeeze()
            if self.subset == 'Test':
                token_neg = f"A photo of a non {age} years old person"
                token_neg = clip.tokenize(token_neg)
                token = torch.cat((token.unsqueeze(0), token_neg), dim=0)
            label_and_attributes = torch.tensor([age, race, gender])
        elif self.label == 'gender':
            gender_str = self.gender_mapping[gender].lower()
            token = f"A photo of a {gender_str} person"
            token = clip.tokenize(token)
            token = token.squeeze()
            if self.subset == 'Test':
                token_neg = f"A photo of a non {gender_str} person"
                # token_neg = f"A photo of a non {gender} person"
                token_neg = clip.tokenize(token_neg)
                token = torch.cat((token.unsqueeze(0), token_neg), dim=0)
            label_and_attributes = torch.tensor([gender, age, race])
        elif self.label == 'race':
            race_str = self.race_mapping[race].lower()
            token = f"A photo of a {race_str} person"
            token = clip.tokenize(token)
            token = token.squeeze()
            if self.subset == 'Test':
                token_neg = f"A photo of a non {race_str} person"
                # token_neg = f"A photo of a non {gender} person"
                token_neg = clip.tokenize(token_neg)
                token = torch.cat((token.unsqueeze(0), token_neg), dim=0)
            label_and_attributes = torch.tensor([race, gender, age])

        return image, token, label_and_attributes


class EuroSat(torch.utils.data.Dataset):

    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.split = split
        assert split in ['train', 'test',
                         'val'], 'Split must be either train, test or val'
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.ToTensor()

        self.templates = [
            lambda c: f'A centered satellite photo of {c}.',
            lambda c: f'A centered satellite photo of a {c}.',
            lambda c: f'A centered satellite photo of the {c}.',
        ]

        self.single_template = lambda c: f'A centered satellite photo of a {c}'

        if not os.path.exists(root + '/DONE'):
            print('Preparing dataset...', file=sys.stderr)
            r = requests.get(
                'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(root)
            os.system(f'mv {root}/EuroSAT_RGB/* {root}')
            os.system(f'rmdir {root}/EuroSAT_RGB')

            # create DONE file
            with open(self.root + '/DONE', 'w') as f:
                f.write('')

            # downlaod split file form https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=self.root + '/split.json')

            print('Done', file=sys.stderr)

        self.data_split = pd.DataFrame(
            json.load(open(self.root + '/split.json', 'r'))[split])
        self.class_names = self.get_class_names()

        self.data = self.data_split[0].values
        self.targets = self.data_split[1].values

        self.prompts = [
            f"A centered satellite photo of {c}" for c in self.class_names]

    @staticmethod
    def get_class_names():
        base_path = "/work/debiasing/frinaldi/mammoth"
        if not os.path.exists(base_path + f'eurosat/DONE'):
            gdd.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                dest_path=base_path + 'eurosat/split.json')
        return pd.DataFrame(json.load(open(base_path + 'eurosat/split.json', 'r'))['train'])[2].unique()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(self.root + '/' + img).convert('RGB')

        not_aug_img = self.totensor(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     if self.split == 'train':
        #         token = clip.tokenize(f"A centered satellite photo of {self.class_names[target].lower()}").squeeze()
        #     else:
        #         token = clip.tokenize([f"A centered satellite photo of {c}" for c in self.class_names])

        return img, target


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=2):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )
        self.train_dataset.class_names = self.train_dataset.classes

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )
        self.test_dataset.class_names = self.test_dataset.classes

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.class_names = self.test_dataset.classes

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

        '''self.class_mapping = {
            0: 'apple',
            1: 'aquarium fish',
            2: 'baby',
            3: 'bear',
            4: 'beaver',
            5: 'bed',
            6: 'bee',
            7: 'beetle',
            8: 'bicycle',
            9: 'bottle',
            10: 'bowl',
            11: 'boy',
            12: 'bridge',
            13: 'bus',
            14: 'butterfly',
            15: 'camel',
            16: 'can',
            17: 'castle',
            18: 'caterpillar',
            19: 'cattle',
            20: 'chair',
            21: 'chimpanzee',
            22: 'clock',
            23: 'cloud',
            24: 'cockroach',
            25: 'couch',
            26: 'crab',
            27: 'crocodile',
            28: 'cup',
            29: 'dinosaur',
            30: 'dolphin',
            31: 'elephant',
            32: 'flatfish',
            33: 'forest',
            34: 'fox',
            35: 'girl',
            36: 'hamster',
            37: 'house',
            38: 'kangaroo',
            39: 'keyboard',
            40: 'lamp',
            41: 'lawn mower',
            42: 'leopard',
            43: 'lion',
            44: 'lizard',
            45: 'lobster',
            46: 'man',
            47: 'maple tree',
            48: 'motorcycle',
            49: 'mountain',
            50: 'mouse',
            51: 'mushroom',
            52: 'oak tree',
            53: 'orange',
            54: 'orchid',
            55: 'otter',
            56: 'palm tree',
            57: 'pear',
            58: 'pickup truck',
            59: 'pine tree',
            60: 'plain',
            61: 'plate',
            62: 'poppy',
            63: 'porcupine',
            64: 'possum',
            65: 'rabbit',
            66: 'raccoon',
            67: 'ray',
            68: 'road',
            69: 'rocket',
            70: 'rose',
            71: 'sea',
            72: 'seal',
            73: 'shark',
            74: 'shrew',
            75: 'skunk',
            76: 'skyscraper',
            77: 'snail',
            78: 'snake',
            79: 'spider',
            80: 'squirrel',
            81: 'streetcar',
            82: 'sunflower',
            83: 'sweet pepper',
            84: 'table',
            85: 'tank',
            86: 'telephone',
            87: 'television',
            88: 'tiger',
            89: 'tractor',
            90: 'train',
            91: 'trout',
            92: 'tulip',
            93: 'turtle',
            94: 'wardrobe',
            95: 'whale',
            96: 'willow tree',
            97: 'wolf',
            98: 'woman',
            99: 'worm'
        }
        '''
