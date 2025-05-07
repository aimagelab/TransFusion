from collections import OrderedDict
from copy import deepcopy
import os
from typing import Optional, Tuple, Union

import torch
import torch.backends.cuda
import torch.backends.cudnn
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torchmetrics import Accuracy
from tqdm import tqdm
from src_.data import DataModule
from src_.model import ClassificationModel
from pytorch_lightning import seed_everything
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torch import nn
import numpy as np 
import types
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                # "trainer.logger": lazy_instance(WandbLogger, project="vit-b16-finetuning-cifar10"),
                "model_checkpoint.monitor": "val_acc",
                "model_checkpoint.mode": "max",
                "model_checkpoint.filename": "best-step-{step}-{val_acc:.4f}",
                "model_checkpoint.save_last": True,
            }
        )
        parser.link_arguments("data.size", "model.image_size")
        parser.link_arguments(
            "data.num_classes", "model.n_classes", apply_on="instantiate"
        )
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os

class DynamicCheckpointDirCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        # Find the ModelCheckpoint callback
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # Ensure the logger is a WandbLogger
                if isinstance(trainer.logger, WandbLogger):
                    run_name = trainer.logger.experiment.name  # Get WandB run name
                    
                    # Use the existing dirpath as the base path
                    if callback.dirpath is None:
                        print("Warning: ModelCheckpoint.dirpath is not set. Using default path.")
                        base_dir = "/default/path/to/checkpoints"  # Fallback path
                    else:
                        base_dir = callback.dirpath
                    
                    # Set the new dirpath with the WandB run name
                    callback.dirpath = os.path.join(base_dir, run_name)
                    os.makedirs(callback.dirpath, exist_ok=True)  # Ensure directory exists
                    print(f"Updated Checkpoint Directory: {callback.dirpath}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set the seed for reproducibility
seed_everything(2, workers=True)

cli = MyLightningCLI(
    ClassificationModel,
    DataModule,
    save_config_kwargs={"overwrite": True},
    trainer_defaults={
        "logger" : lazy_instance(WandbLogger, project="vit-b16-finetuning-cifar10"),
        "check_val_every_n_epoch": None,
        "callbacks": [DynamicCheckpointDirCallback()], # Add the custom callback
    }
)

            
# Copy the config into the experiment directory
# Fix for https://github.com/Lightning-AI/lightning/issues/17168
try:
    os.rename(
        os.path.join(cli.trainer.logger.save_dir, "config.yaml"),  # type:ignore
        os.path.join(
            cli.trainer.checkpoint_callback.dirpath[:-12], "config.yaml"  # type:ignore
        ),
    )
except:
    pass
