from PIL import Image
import torchvision.transforms as transforms
import sys
import zipfile
import requests
import torch
import os
import json
import pandas as pd
import io
from typing import Tuple

try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except ImportError:
    raise ImportError(
        "Please install the google_drive_downloader package by running: `pip install googledrivedownloader`")


class EuroSat(torch.utils.data.Dataset):

    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.split = split
        assert split in ['train', 'test','val'], 'Split must be either train, test or val'
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
        self.train_dataloader = None
        self.test_dataloader = None


    @staticmethod
    def get_class_names():
        # base_path = "/leonardo_scratch/large/userexternal/frinaldi/datasets/"
        base_path = "/work/debiasing/datasets/"
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

        #not_aug_img = self.totensor(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target
