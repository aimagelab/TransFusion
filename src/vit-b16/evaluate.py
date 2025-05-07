from collections import OrderedDict
from copy import deepcopy
import os
from typing import Optional, Tuple, Union
import csv
import torch
import torch.backends.cuda
import torch.backends.cudnn
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torchmetrics import Accuracy
from tqdm import tqdm
import sys
sys.path.append('/homes/frinaldi/Merge2Update')
from permutations.permutation_spec import ViT_B_PermutationSpecBuilder, Naive_ViT_B_PermutationSpecBuilder
from permutations.weights_matcher import WeightMatcher, ProcustesWeightMatcher, LayerIterationOrder
from permutations.utils import apply_permutation_to_statedict 
from src_.data import DataModule
from src_.model import ClassificationModel
from pytorch_lightning import seed_everything
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torch import nn
import numpy as np 
import types
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser
from run import svd_decompose_state_dict

# Initialize a figure for plotting
from matplotlib import pyplot as plt

def evaluate_vit(dataloader, model):
    eval_avg_loss = 0
    all_probs = []
    all_targets = []
    ce_loss = nn.CrossEntropyLoss()
    accuracy = Accuracy(task='multiclass', num_classes=10).to(device)
    model.eval()
    for batch in tqdm(dataloader):
        images, targets = batch
        images = images.to(device)
        targets =  targets.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            loss = ce_loss(outputs, targets)
        
            eval_avg_loss += loss.item()  
    all_probs = torch.cat(all_probs, axis=0)
    all_targets = torch.cat(all_targets, axis=0)
    eval_avg_loss /= len(dataloader)  
    acc = accuracy(all_probs, all_targets)
    return acc, eval_avg_loss

def loss_barrier(model_a, model_b, dataloader, alphas=np.linspace(0, 1, 11)):
    """
    Evaluate and plot the loss barrier between model_a and model_b.
    
    Parameters:
        model_a: The first model.
        model_b: The second model.
        dataloader: Dataloader for evaluation.
        alphas: Array of interpolation coefficients to evaluate.

    Returns:
        losses: List of losses for the interpolated models.
    """
    # Prepare a deepcopy of model_a to hold the interpolated weights
    model_c = deepcopy(model_a)
    model_c = model_c.to(device)

    # Extract the state dictionaries
    theta_a = model_a.net.state_dict()
    theta_b = model_b.net.state_dict()

    losses = []
    accs =[]
    for alpha in tqdm(alphas, desc="Evaluating Loss Barrier"):
        # Interpolate weights
        theta_c = interpolate(theta_a, theta_b, alpha)
        model_c.net.load_state_dict(theta_c)

        # Evaluate interpolated model
        acc , loss = evaluate_vit(dataloader, model_c)
        losses.append(loss)
        accs.append(acc.cpu().numpy())

    return alphas, accs, losses
def interpolate(theta_a: OrderedDict, theta_b: OrderedDict, alpha = 0.5) -> OrderedDict:
    assert set(theta_a.keys()) == set(theta_b.keys())
    theta = {}
    for key in theta_a.keys():
        if 'identity' in key:
            sim_mat = (1-alpha) * theta_a[key] + alpha * theta_b[key]
            # sim_mat_normalized = sinkhorn_normalization(sim_mat)
            row_ind, col_ind = linear_sum_assignment(sim_mat.cpu().numpy(), maximize=True)
            I3 = torch.zeros_like(sim_mat).to(device)
            I3[row_ind, col_ind] = 1
            theta[key] = I3
            # Compute the Frobenius norm
            # frobenius_norm = torch.norm(I3 - sim_mat, p='fro')**2
            # print(f"Frobenius norm ||I3 - [(1-alpha)I1 + alphaI2] |_F^2: {frobenius_norm.item()}")
            
        else:
            theta[key] = (1-alpha) * theta_a[key] + alpha * theta_b[key]

    return theta
def my_forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    
    self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    # first residual connection
    hidden_states = attention_output + self.attention.shortcut_1(hidden_states)

    # in ViT, layernorm is also applied after self-attention
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.intermediate(layer_output)

    # second residual connection is done here
    layer_output = self.output(layer_output, self.attention.shortcut_2(hidden_states))

    outputs = (layer_output,) + outputs

    return outputs


import copy
def add_interpolated_block(model, index_A, index_B):
    a = model.net.vit.encoder.layer[index_A]
    b = model.net.vit.encoder.layer[index_B]

    # Deep copy block A to create a new interpolated block
    new_block = copy.deepcopy(a)

    for (name_a, param_a), (name_b, param_b) in zip(a.named_parameters(), b.named_parameters()):
        param_new = 0.5 * param_a.data + 0.5 * param_b.data
        dict(new_block.named_parameters())[name_a].data.copy_(param_new)

    # # Interpolate buffers (e.g., layer norm running stats)
    # for (name_a, buffer_a), (name_b, buffer_b) in zip(a.named_buffers(), b.named_buffers()):
    #     buffer_new = 0.5 * buffer_a.data + 0.5 * buffer_b.data
    #     dict(new_block.named_buffers())[name_a].data.copy_(buffer_new)

    # Insert the interpolated block
    block_list = list(model.net.vit.encoder.layer)
    insert_index = min(index_A, index_B)
    block_list.insert(insert_index, new_block)
    
    model.net.vit.encoder.config.num_hidden_layers = len(block_list)

    # Update the model
    model.net.vit.encoder.layer = nn.ModuleList(block_list)
    
def sub_interpolated_block(model, index_A, index_B):
    a = model.net.vit.encoder.layer[index_A]
    b = model.net.vit.encoder.layer[index_B]

    # Deep copy block A to create a new interpolated block
    new_block = copy.deepcopy(a)

    for (name_a, param_a), (name_b, param_b) in zip(a.named_parameters(), b.named_parameters()):
        param_new = 0.5 * param_a.data + 0.5 * param_b.data
        dict(new_block.named_parameters())[name_a].data.copy_(param_new)

    # # Interpolate buffers (e.g., layer norm running stats)
    # for (name_a, buffer_a), (name_b, buffer_b) in zip(a.named_buffers(), b.named_buffers()):
    #     buffer_new = 0.5 * buffer_a.data + 0.5 * buffer_b.data
    #     dict(new_block.named_buffers())[name_a].data.copy_(buffer_new)

    # Insert the interpolated block
    block_list = list(model.net.vit.encoder.layer)
    block_list[index_A] = new_block
    block_list.pop(index_B)
    
    model.net.vit.encoder.config.num_hidden_layers = len(block_list)

    # Update the model
    model.net.vit.encoder.layer = nn.ModuleList(block_list)

    
    

def add_shortcuts(model):
    for layer in model.net.vit.encoder.layer:
        layer.forward = types.MethodType(my_forward, layer)
        layer.attention.shortcut_1 = Shortcut(model.net.vit.encoder.config.hidden_size).to(device)
        layer.attention.shortcut_2 = Shortcut(model.net.vit.encoder.config.hidden_size).to(device)

    
class Shortcut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: should have requires_grad=False
        self.identity = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.identity.T
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_a", type=str, default="/work/debiasing/frinaldi/vit-b-finetuning/cifar10/kind-snowball-26/best-step-step=36660-val_acc=0.7900.ckpt", help="Path to first model checkpoint")
    parser.add_argument("--ckpt_b", type=str, default="/work/debiasing/frinaldi/vit-b-finetuning/cifar10/cool-frost-25/best-step-step=36660-val_acc=0.7650.ckpt", help="Path to second model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="/work/debiasing/datasets", help="Path to evaluation dataset")
    parser.add_argument("--output_path", type=str,default="/homes/frinaldi/Merge2Update/plots/", help="Path to save results")
    parser.add_argument("--train_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--interp_pts", type=int, default=5, help="Number of interpolation points")
    
    args = parser.parse_args()
    

    preprocess = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    test_dataset = CIFAR10(root=args.dataset_path, download=True, train=False, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model_a = ClassificationModel(weights=args.ckpt_a).to(device)
    
    for i in range(1):
        num_layers = model_a.net.vit.encoder.config.num_hidden_layers
        # add_shortcuts(model_a)
        sub_interpolated_block(model_a, 0,1)
    

    
    # Evaluate interpolated model
    acc , loss = evaluate_vit(test_loader, model_a)
    print(f"Accuracy: {acc} , Loss: {loss}")
    new_sd = svd_decompose_state_dict(model_a.state_dict(), 0.99)
    model_a.load_state_dict(new_sd)
    acc , loss = evaluate_vit(test_loader, model_a)
    print(f"SVD Accuracy: {acc} , Loss: {loss}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    main()

