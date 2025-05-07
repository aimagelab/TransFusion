import os
import torch

import abc
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader
import yaml 
#import templates
from .templates import get_templates
from PIL import Image
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random

def smart_joint(*paths):
    return os.path.join(*paths).replace("\\", "/")


# modified from: https://github.com/microsoft/torchgeo
class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and labels at that index
        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    """Abstract base class for classification datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialize a new VisionClassificationDataset instance.
        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

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

        return img, target

    def __len__(self):
        return len(self.targets)


class RESISC45Dataset(VisionClassificationDataset):

    directory = "resisc45/NWPU-RESISC45"
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]
    # download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar

    def __init__(self, root: str = "data", split: str = "train",transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None) -> None:
        """Initialize a new RESISC45 dataset instance.
        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = os.path.join(root, 'NWPU-RESISC45')
        self.transform = transforms
        self.target_transform = None

        if split == "train":
            data_config = yaml.load(open(smart_joint(self.root, 'resisc45_train.yaml')), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(smart_joint(self.root, 'resisc45_test.yaml')), Loader=yaml.Loader)

        self.data = np.array([smart_joint(self.root, d) for d in data_config['data']])
        self.targets = np.array(data_config['targets']).astype(np.int16)

class RESISC45:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 few_shot=False):

        self.train_dataset = RESISC45Dataset(root=location, split='train', transforms=preprocess)
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

        self.test_dataset = RESISC45Dataset(root=location, split='test', transforms=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # class names have _ so split on this for better zero-shot head
        self.class_names = [' '.join(c.split('_')) for c in RESISC45Dataset.classes]
        self.templates = get_templates('resisc45')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'A photo of a {c}'
