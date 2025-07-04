"""
Utility functions for environment setup and model loading.
Includes reproducibility setup and model loading helpers.
"""
######################################################################
# Environment and Model Utilities
######################################################################
from collections import defaultdict
from copy import deepcopy
import torch
import numpy as np
import random
import open_clip
# from src.datasets import EuroSat, CIFAR100
from src.dataset.eurosat import EuroSat
from src.dataset.cifar100 import CIFAR100
from src.dataset.sun397 import SUN397
from src.dataset.dtd import DTD
from src.dataset.svhn import SVHN
from src.dataset.gtsrb import GTSRB
from src.dataset.resisc45 import RESISC45
from src.dataset.imagenet_r import IMAGENETR
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import torch.nn as nn
from src.modules import accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments for CLIP zero-shot evaluation and permutation experiments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='TransFusion: CLIP Zero-shot Evaluation and Permutation Transfer')
    parser.add_argument('--seed', default=89, type=int,
                        help='Seed for reproducibility.')
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['cifar100', 'eurosat', 'dtd', 'gtsrb', 'resisc45', 'svhn', 'sun397', 'imagenetr'], help="Dataset to evaluate.")
    parser.add_argument('--batch_size', default=32,
                        type=int, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers for data loading.')
    parser.add_argument('--arch', default='ViT-B-16',
                        type=str, help='Model architecture.')
    parser.add_argument('--pretraining_backbone_A', default='commonpool_l_s1b_b8k',
                        type=str, help='Pretraining model A for backbone1.')
    parser.add_argument('--pretraining_backbone_B', default='datacomp_l_s1b_b8k',
                        type=str, help='Pretraining model B for backbone2.')
    parser.add_argument('--finetuned_checkpoint_A', default=None,
                        type=str, help='Path to finetuned model A. If not set, defaults to checkpoints/<arch>/<dataset>/model.pt')
    parser.add_argument('--alpha', default=0.8, type=float,
                        help='Scaling coefficient.')
    parser.add_argument('--experiment_name', default='TransFusion', type=str, help='Experiment name.')
    parser.add_argument('--max_alpha', default=1,
                        type=float, help='Max alpha.')
    parser.add_argument('--wandb_mode', default='offline', type=str,
                        choices=['online', 'offline', 'disabled'], help='Wandb mode')
    parser.add_argument('--wandb_project', default='TransFusion',
                        type=str, help='Wandb project name')
    parser.add_argument(
        '--base_folder', default='/work/debiasing/')
    args = parser.parse_args()
    if args.finetuned_checkpoint_A is None:
        args.finetuned_checkpoint_A = f"checkpoints/{args.arch}/{args.dataset}/model.pt"
    return args


def setup_environment(args):
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Device configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def get_models(args, device):
    backbone_A, _, preprocess = open_clip.create_model_and_transforms(args.arch,
                                                                      pretrained=args.pretraining_backbone_A,
                                                                      cache_dir=f'{args.base_folder}/open_clip',
                                                                      device=device)
    backbone_B, _, _ = open_clip.create_model_and_transforms(args.arch,
                                                             pretrained=args.pretraining_backbone_B,
                                                             cache_dir=f'{args.base_folder}/open_clip',
                                                             device=device)
    state_dict = torch.load(args.finetuned_checkpoint_A)['model_state_dict']
    model_A_ft = deepcopy(backbone_A)
    model_A_ft.load_state_dict(state_dict)

    return backbone_A, backbone_B, model_A_ft, preprocess


def load_dataset(args, preprocess, support=False, few_shot=False):
    if support:
        logger.info("Loading target and support dataset...")
        if args.dataset == 'cifar100':
            target_dataset = CIFAR100(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'eurosat':
            target_dataset = EuroSat(
                root=f"{args.base_folder}/datasets/eurosat", split='test', transform=preprocess)
            target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=True, drop_last=False)
        elif args.dataset == 'sun397':
            target_dataset = SUN397(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'dtd':
            target_dataset = DTD(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'svhn':
            target_dataset = SVHN(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'gtsrb':
            target_dataset = GTSRB(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'resisc45':
            target_dataset = RESISC45(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            target_dataloader = target_dataset.test_loader
        elif args.dataset == 'imagenetr':
            target_dataset = IMAGENETR(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers)
            target_dataloader = target_dataset.test_loader

        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        logger.info(f'Number of target samples: {len(target_dataloader.dataset)}')

        # support_dataset = ImageNet(preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers)
        # support_dataset = CIFAR100(preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
        support_dataset = IMAGENETR(
            preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers)
        support_dataloader = support_dataset.test_loader

        logger.info(f'Number of support samples: {len(support_dataloader.dataset)}')

        return target_dataloader, target_dataset, support_dataloader, support_dataset

    else:
        logger.info("Loading dataset...")
        if args.dataset == 'cifar100':
            dataset = CIFAR100(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)

            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'eurosat':
            test_dataset = EuroSat(
                root=f"{args.base_folder}/datasets/eurosat", split='test', transform=preprocess)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True, drop_last=False)

            dataset = EuroSat(
                root=f'{args.base_folder}/datasets/eurosat', split='train', transform=preprocess)
            if few_shot:
                # Sample few examples per class for training
                class_indices = defaultdict(list)
                for idx, (_, label) in enumerate(dataset):
                    class_indices[label].append(idx)

                # Limit to 'samples_per_class' per class
                sampled_indices = []
                for indices in class_indices.values():
                    sampled_indices.extend(random.sample(
                        indices, min(10, len(indices))))

                train_dataset_subset = Subset(dataset, sampled_indices)
                train_loader = DataLoader(
                    train_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            else:
                train_loader = DataLoader(
                    dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)
        elif args.dataset == 'sun397':
            dataset = SUN397(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'dtd':
            dataset = DTD(preprocess=preprocess,
                          location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'svhn':
            dataset = SVHN(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'gtsrb':
            dataset = GTSRB(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'resisc45':
            dataset = RESISC45(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers, batch_size=args.batch_size)
            test_loader = dataset.test_loader
            train_loader = dataset.train_loader
        elif args.dataset == 'imagenetr':
            dataset = IMAGENETR(
                preprocess=preprocess, location=f'{args.base_folder}/datasets', num_workers=args.workers)
            test_loader = dataset.test_loader
            train_loader = None
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        return train_loader, test_loader, dataset


def evaluate_model(model, dataloader, dataset, device='cuda:0', prompt_ensemble=True, first_n_batches=None):
    eval_avg_loss = 0
    all_probs = []
    all_labels = []
    ce_loss = nn.CrossEntropyLoss()
    model.eval()

    if prompt_ensemble:
        # prompts =  [[template(c) for c in cifar.class_names] for template in cifar.templates] #cifar100
        prompts = [[template(c.lower()) for c in dataset.class_names]
                   for template in dataset.templates]
        with torch.no_grad():
            template_embeddings = []
            for template in prompts:
                test_texts = open_clip.tokenize(template)
                test_texts = test_texts.to(device)
                test_text_features = F.normalize(
                    model.encode_text(test_texts), dim=-1)
                template_embeddings.append(test_text_features)

            text_features = torch.mean(torch.stack(template_embeddings), dim=0)
    else:
        prompts = [dataset.single_template(c.lower())
                   for c in dataset.class_names]

        with torch.no_grad():
            test_texts = open_clip.tokenize(prompts)
            test_texts = test_texts.to(device)
            text_features = F.normalize(model.encode_text(test_texts), dim=-1)
    for id, batch in tqdm(enumerate(dataloader)):
        if first_n_batches is not None:
            if id == first_n_batches:
                break
        images, targets = batch

        images = images.to(device)

        targets = targets.to(device)

        targets = targets.long()  # fix resisc45

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = F.normalize(model.encode_image(images), dim=-1)
            vl_logits = 100 * \
                (torch.einsum('ij,cj->ic', image_features, text_features))

        vl_prob = torch.softmax(vl_logits.float(), dim=-1)

        all_probs.append(vl_prob.cpu().numpy())
        all_labels.append(targets.cpu().numpy())
        # all_attrs.append(attributes.cpu().numpy())
        loss = ce_loss(vl_logits, targets)

        eval_avg_loss += loss.item()

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    eval_avg_loss /= len(dataloader)

    overall_acc = accuracy(all_probs, all_labels, topk=(1,))
    return eval_avg_loss, overall_acc


def evaluate_target_and_support(model, dataloaders: list, datasets: list, device, prompt_ensemble=True) -> dict:
    results = {}
    for dataloader, dataset in zip(dataloaders, datasets):
        loss, accuracy = evaluate_model(
            model, dataloader, dataset, device, prompt_ensemble)
        results[dataset.__class__.__name__] = (loss, accuracy)
    return results
