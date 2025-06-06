"""
Script for CLIP model training and fine-tuning with logging and evaluation.
Handles argument parsing, training loop, evaluation, and logging.
"""
######################################################################
# CLIP Training and Fine-Tuning Script
######################################################################
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from utils import load_dataset
from torchmetrics.classification import Accuracy as torch_Accuracy
from open_clip.modified_resnet import AttentionPool2d
import lightning as L
import open_clip
from torch.utils.data import Subset
import logging
from tqdm import tqdm
from src.modules import *
import os
from pathlib import Path
import numpy as np
import random
import argparse
import time
import json
from collections import Counter
import clip

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import sys
sys.path.append('.')

try:
    import wandb
    import wandbbq
    os.environ["WANDB__SERVICE_WAIT"] = "800"
except ImportError:
    wandb = None

parser = argparse.ArgumentParser(description='CLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=33, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--num_steps', default=2000, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')
# parser.add_argument('--log_level', default=logging.INFO)
parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--model_arch', default='ViT-B-16',
                    type=str, help='options: ViT-B-16 | VIT-B-32 | VIT-L-14')
parser.add_argument('--pretraining', default='openai',
                    type=str, help='pretraining dataset to use')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--dataset', default='resisc45', type=str,
                    help='Options: eurosat, cifar100, sun397, dtd, svhn, gtsrb, resisc45, imagenetr')


parser.add_argument("--wandb_project", type=str,
                    default='', help="Wandb project name")
parser.add_argument("--wandb_mode", type=str,
                    default='offline', help="Wandb mode")
parser.add_argument("--wandb_run_name", type=str,
                    default='', help="Wandb run name")

parser.add_argument('--base_folder', type=str,
                    default="/work/debiasing/frinaldi", help='Base folder to store models.')


class LiTCLIP(L.LightningModule):
    def __init__(self, clip_model, dataset, tokenizer=clip.tokenize, prompt_ensemble=True):
        super().__init__()
        self.clip_model = clip_model
        self.loss = nn.CrossEntropyLoss()
        self.best_epoch = 0
        self.best_loss = 1000000
        self.best_acc = sys.float_info.min
        self.ce_loss = nn.CrossEntropyLoss()
        self.prompt_ensemble = prompt_ensemble
        self.dataset = dataset
        self.accuracy = torch_Accuracy(
            task="multiclass", num_classes=len(self.dataset.class_names))
        self.tokenizer = tokenizer

        self.train_loss_sum = 0.0
        self.train_steps = 0
        self.best_step = 0

        self.save_hyperparameters()

    # def on_train_epoch_start(self):
    #     self.avg_loss = 0

    def on_train_start(self):
        self.logger.watch(self.clip_model, log='all', log_freq=10)

    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)
        texts = self.tokenizer([self.dataset.single_template(self.dataset.class_names[i.item(
        )].lower()) for i in labels])  # single template -> A photo of/ A centered photo of
        texts = texts.to(device)

        logits_per_image, logits_per_text = self.clip_model.get_logits(
            images, texts)

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)

        total_loss = (self.loss(logits_per_image, ground_truth) +
                      self.loss(logits_per_text, ground_truth))/2
        self.train_loss_sum += total_loss.item()
        self.train_steps += 1
        current_lr = self.lr_schedulers().get_last_lr(
        )[-1] if self.lr_schedulers() else self.hparams.lr
        total_norm = 0
        for p in self.clip_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5  # L2 norm

        # Log metrics
        self.log('train/loss', total_loss, on_step=True)
        self.log('train/lr', current_lr, on_step=True)
        self.log('train/grad_norm', total_norm, on_step=True)

        return total_loss

    # def on_train_epoch_end(self):
    #     self.avg_loss /= len(self.trainer.train_dataloader)
    #     wandb.log({'Avg train loss' : self.avg_loss, 'lr' : self.lr_schedulers().get_last_lr()[-1], 'Epoch' : self.current_epoch,})
    #     print(f'Epoch: {self.current_epoch}')
    #     print(f'Avg training loss: {self.avg_loss}')

    def on_validation_start(self):
        if self.train_steps > 0:
            avg_train_loss = self.train_loss_sum / self.train_steps
            current_lr = self.lr_schedulers().get_last_lr(
            )[-1] if self.lr_schedulers() else self.hparams.lr
            self.logger.experiment.log({
                'train/avg_loss': avg_train_loss,
                'train/lr': current_lr
            }, step=self.global_step)
            print(
                f'Step {self.global_step}: Avg training loss: {avg_train_loss}')
            self.train_loss_sum = 0.0
            self.train_steps = 0

        self.eval_avg_loss = 0
        self.all_probs = []
        self.all_labels = []
        self.ce_loss = nn.CrossEntropyLoss()
        if self.prompt_ensemble:
            prompts = [[template(c.lower()) for c in self.dataset.class_names]
                       for template in self.dataset.templates]  # eurosat
            with torch.no_grad():
                template_embeddings = []
                for template in prompts:
                    test_texts = self.tokenizer(template)
                    test_texts = test_texts.to(self.device)
                    test_text_features = F.normalize(
                        self.clip_model.encode_text(test_texts), dim=-1)
                    template_embeddings.append(test_text_features)

                self.text_features = torch.mean(
                    torch.stack(template_embeddings), dim=0)
        else:
            prompts = [self.dataset.single_template(
                c.lower()) for c in self.dataset.class_names]
            with torch.no_grad():
                test_texts = self.tokenizer(prompts)
                test_texts = test_texts.to(self.device)
                self.text_features = F.normalize(
                    self.clip_model.encode_text(test_texts), dim=-1)

    def validation_step(self, batch, batch_idx):

        images, targets = batch

        images = images.to(self.device)

        targets = targets.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = F.normalize(
                self.clip_model.encode_image(images), dim=-1)
            vl_logits = 100 * \
                (torch.einsum('ij,cj->ic', image_features, self.text_features))

        vl_prob = torch.softmax(vl_logits.float(), dim=-1)

        self.all_probs.append(vl_prob.cpu().numpy())
        self.all_labels.append(targets.cpu().numpy())

        targets = targets.long()  # fix resisc45

        loss = self.ce_loss(vl_logits, targets)

        self.eval_avg_loss += loss.item()

    def on_validation_end(self):
        self.all_probs = np.concatenate(self.all_probs, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)
        self.eval_avg_loss /= len(self.trainer.val_dataloaders)

        # overall_acc = accuracy(self.all_probs, self.all_labels, topk=(1,))
        overall_acc = self.accuracy(torch.from_numpy(
            self.all_probs), torch.from_numpy(self.all_labels)).item()
        # if self.trainer.state.stage != "sanity_check":

        # Log validation metrics
        self.logger.experiment.log({'val/acc': overall_acc,
                                   'val/loss': self.eval_avg_loss})

        print(
            f'Step {self.global_step}: Eval accuracy: {overall_acc}, Avg eval loss: {self.eval_avg_loss}')

        # for name, param in self.clip_model.named_parameters():
        #     wandb.log({f'Weights/{name}': wandb.Histogram(param.cpu().detach().numpy())})

        if self.best_acc <= overall_acc:
            self.best_acc = overall_acc
            self.best_step = self.global_step
            torch.save({
                'step': self.global_step,
                'model_state_dict': self.clip_model.state_dict(),
                'best_acc': self.best_acc,
            }, os.path.join(args.result_dir, "best.pt"))

    def on_fit_end(self):
        self.logger.experiment.log({
            'best/acc': self.best_acc,
            'best/step': self.best_step
        })
        print(f'Best Step: {self.best_step}')

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.clip_model.visual.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)

        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: step / 200 if step < 200 else 1.0
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=1800,  # Steps after warmup
            eta_min=0.0   # Minimum learning rate
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[200]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update learning rate every step
                "frequency": 1
            }
        }


if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    if args.wandb_run_name != '':
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=args.base_folder,
            mode=args.wandb_mode,
            config=vars(args)
        )
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            save_dir=args.base_folder,
            mode=args.wandb_mode,
            config=vars(args)
        )

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    print(f'===> Seed: {args.seed}')

    # If using GPU then use mixed precision training.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_arch,
        pretrained=args.pretraining,
        device=device,
        force_quick_gelu=False,
        cache_dir=f'{args.base_folder}/openclip'
    )
    tokenizer = open_clip.get_tokenizer(args.model_arch)
    # args.base_folder = "/work/debiasing"
    train_dataloader, test_dataloader, dataset = load_dataset(args, preprocess)
    # args.base_folder = "/work/debiasing/frinaldi"
    print(f"Finetuning on {args.dataset} dataset for {args.num_steps} steps")

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)
        # start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) useless since every finetuning is on a different dataset

    lit_model = LiTCLIP(model, dataset=dataset,
                        tokenizer=tokenizer, prompt_ensemble=True)
    lit_model.hparams.update(vars(args))
    trainer = L.Trainer(max_steps=args.num_steps,
                        val_check_interval=250,
                        check_val_every_n_epoch=None,
                        num_sanity_val_steps=-1,
                        logger=logger,
                        default_root_dir=args.base_folder,
                        enable_checkpointing=False,
                        accumulate_grad_batches=4)
    # trainer.validate(model, test_dataloader)
    trainer.fit(lit_model, train_dataloader, test_dataloader)
