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
    
    seed_everything(args.train_seed, workers=True)

    preprocess = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    test_dataset = CIFAR10(root=args.dataset_path, download=True, train=False, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model_a = ClassificationModel(weights=args.ckpt_a).to(device)
    model_b = ClassificationModel(weights=args.ckpt_b).to(device)
    add_shortcuts(model_a)
    add_shortcuts(model_b)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compute the loss barrier for A-B
    alphas, accs, losses = loss_barrier(model_a, model_b, test_loader, alphas=np.linspace(0, 1, args.interp_pts))
    axes[0].plot(alphas, accs, label="Acc Barrier A-B", color="blue", linewidth=2)
    axes[1].plot(alphas, losses, label="Loss Barrier A-B", color="blue", linewidth=2)
    plot_name = "loss_acc_barriers_results_no-residual"
    # Prepare CSV file for writing
    permutation_spec = ViT_B_PermutationSpecBuilder(depth=model_a.net.vit.encoder.config.num_hidden_layers).create_permutation_spec()
    csv_file = f"{args.output_path}/{plot_name}.csv"
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Iteration", "Alpha", "Accuracy", "Loss", "Perm_Accuracy", "Perm_Loss"])
        #Save A-B results
        for alpha, acc, loss in zip(alphas, accs, losses):
            writer.writerow(["A-B", alpha, acc, loss, None, None])
            
        weight_matcher = ProcustesWeightMatcher(
        ps=permutation_spec,
        max_iter=10,
        p_trim=None,
        fixed=model_b.net.state_dict(),
        permutee=model_a.net.state_dict(),
        num_heads=model_a.net.vit.encoder.config.num_attention_heads,
        intra_head=True,
        layer_iteration_order=LayerIterationOrder.FORWARD)
    
        permutation, heads_permutation = weight_matcher.run()
        
        perm_model_a = deepcopy(model_a)
        perm_state_dict = apply_permutation_to_statedict(permutation_spec, permutation, model_a.net.state_dict(), heads_permutation=heads_permutation)
        perm_model_a.net.load_state_dict(perm_state_dict)
        
        # Compute the loss barrier for permA-B
        alphas, perm_accs, perm_losses = loss_barrier(perm_model_a, model_b, test_loader, alphas=np.linspace(0, 1, args.interp_pts))
        
        # Write permuted results to the CSV
        for alpha, perm_acc, perm_loss in zip(alphas, perm_accs, perm_losses):
            print(f"Alpha:{alpha}, perm_acc: {perm_acc}, loss: {perm_loss}")
            writer.writerow([f"PA-B", alpha, None, None, perm_acc, perm_loss])

        axes[0].plot(alphas, perm_accs, label="Acc Barrier PA-B", linestyle="--", alpha=0.7)
        axes[1].plot(alphas, perm_losses, label="Loss Barrier PA-B", linestyle="--", alpha=0.7)
    
    # Add plot details 
    axes[1].set_title("Loss barriers")
    axes[0].set_title("Accuracy barriers")
    axes[1].set_xlabel("Alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Accs")
    axes[1].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].grid(True)
    axes[1].legend()
    plt.savefig(f"{args.output_path}/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    main()

