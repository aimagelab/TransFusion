"""
Classes and functions for aligning and permuting model weights using assignment algorithms.
Implements WeightMatcher and BruteforceWeightMatcher for layer-wise and head-wise permutation matching.
Provides utilities for solving linear assignment problems and computing similarity between model parameters.
"""
######################################################################
# Weight Matching Algorithms
######################################################################
######################################################################
# Utility Functions
######################################################################
import logging
from enum import auto
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import torch
from strenum import StrEnum
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from .permutation_spec import PermutationSpec
from .utils import *


class LayerIterationOrder(StrEnum):
    """
    Enum for specifying the order in which layers are iterated during weight matching.
    """
    RANDOM = auto()
    FORWARD = auto()
    BACKWARD = auto()
    ALTERNATE = auto()


eps = 1e-12


def get_layer_iteration_order(layer_iteration_order: LayerIterationOrder, num_layers: int):
    """
    Returns the order in which to iterate layers based on the specified strategy.
    Args:
        layer_iteration_order (LayerIterationOrder): The iteration strategy.
        num_layers (int): Number of layers.
    Returns:
        Sequence[int]: The order of layer indices.
    """
    if layer_iteration_order == LayerIterationOrder.RANDOM:
        return torch.randperm(num_layers)
    elif layer_iteration_order == LayerIterationOrder.FORWARD:
        return torch.arange(num_layers)
    elif layer_iteration_order == LayerIterationOrder.BACKWARD:
        return range(num_layers)[num_layers:0:-1]
    elif layer_iteration_order == LayerIterationOrder.ALTERNATE:
        return alternate_layers(num_layers)
    else:
        raise NotImplementedError(
            f"Unknown layer iteration order {layer_iteration_order}")


def alternate_layers(num_layers):
    """
    Returns a list of layer indices in an alternating order (start, end, next, prev, ...).
    Args:
        num_layers (int): Number of layers.
    Returns:
        List[int]: Alternating order of indices.
    """
    all_layers = list(range(num_layers))
    result = []

    for i in range((num_layers + 1) // 2):
        result.append(all_layers[i])
        if i != num_layers - i - 1:
            result.append(all_layers[-i - 1])

    return result


class WeightMatcher:
    """
    Aligns and permutes model weights by solving assignment problems for each layer or head.
    Supports intra-head and extra-head matching, normalization, and pruning.

    Args:
        ps (PermutationSpec): The permutation specification describing how permutations map to layers and axes.
        fixed (Dict[str, Tensor]): State dict of the reference (fixed) model to align to.
        permutee (Dict[str, Tensor]): State dict of the model to be permuted/aligned.
        max_iter (int): Maximum number of matching iterations (default: 100).
        init_perm (Dict[str, Tensor], optional): Initial permutation indices for each permutation group (default: None).
        layer_iteration_order (LayerIterationOrder): Order in which to iterate layers during matching (default: FORWARD).
        p_trim (float, optional): If set, quantile threshold for pruning small weights (default: None).
        num_heads (int, optional): Number of attention heads (for attention layers, default: None).
        intra_head (bool): Whether to perform intra-head matching for attention layers (default: False).
        normalize_weights (bool): Whether to normalize weights before computing similarity (default: False).
    """

    def __init__(self,
                 ps: PermutationSpec,
                fixed: Dict[str, Tensor],
                permutee: Dict[str, Tensor],
                max_iter=100,
                init_perm: Dict[str, Tensor] = None,
                layer_iteration_order: LayerIterationOrder = LayerIterationOrder.FORWARD,
                p_trim: float = None,
                num_heads: int = None,
                intra_head: bool = False,
                normalize_weights: bool = False):
        self.ps = ps
        self.params_a = fixed
        self.params_b = permutee
        self.max_iter = max_iter
        self.init_perm = init_perm
        self.layer_iteration_order = layer_iteration_order
        self.p_trim = p_trim
        self.num_heads = num_heads
        self.intra_head = intra_head
        self.normalize_weights = normalize_weights

        self.perm_sizes = {p: self.params_a[ref_tuple[0]].shape[ref_tuple[1]]
                for p, params_and_axes in ps.perm_to_layers_and_axes.items()
                    for ref_tuple in [params_and_axes[0]]
        }
        self.all_perm_indices = self._initialize_perm_indices()
        self.all_heads_indices = self._initialize_head_indices()

        self.perm_names = list(self.all_perm_indices.keys())
        self.num_layers = len(self.perm_names)

    def _initialize_perm_indices(self) -> Dict[str, Tensor]:
        """
        Initializes permutation indices for each permutation group.
        Returns:
            Dict[str, Tensor]: Mapping from permutation name to indices.
        """
        if self.num_heads is not None:
            return (
                {
                    p: torch.arange(n) if "attn" not in p else torch.arange(
                        n // (n // self.num_heads))
                    for p, n in self.perm_sizes.items()
                }
                if self.init_perm is None
                else self.init_perm
            )
        else:
            return {p: torch.arange(n) for p, n in self.perm_sizes.items()} if self.init_perm is None else self.init_perm

    def _initialize_head_indices(self) -> Union[Dict[str, Dict[str, Tensor]], None]:
        """
        Initializes intra-head permutation indices if intra_head is enabled.
        Returns:
            Dict[str, Dict[str, Tensor]] or None: Mapping from permutation name to head indices.
        """
        if self.intra_head:
            return {
                key: {f"P_head_{idx}": torch.arange(
                    n // self.num_heads) for idx in range(self.num_heads)}
                for key, n in zip(self.all_perm_indices.keys(), self.perm_sizes.values())
                if "attn" in key
            }
        else:
            return None

    def _initialize_similarity_matrices(self, p: str, intra_head_phase: bool):
        """
        Initializes similarity matrices for a given permutation group, handling attention and non-attention cases.
        Args:
            p (str): Permutation name.
            intra_head_phase (bool): Whether intra-head matching is being performed.
        Returns:
            Tuple[int, Tensor or Dict[str, Tensor]]: Number of neurons and similarity matrix/matrices.
        """
        if "attn" in p and self.num_heads is not None:
            if intra_head_phase:
                num_neurons = self.perm_sizes[p] // self.num_heads
                return num_neurons, {f"sim_matrix_{idx}": torch.zeros((num_neurons, num_neurons)).cuda() for idx in range(self.num_heads)}
            else:
                num_neurons = self.perm_sizes[p] // (
                    self.perm_sizes[p] // self.num_heads)
                return num_neurons, torch.zeros((num_neurons, num_neurons)).cuda()
        else:
            num_neurons = self.perm_sizes[p]
            return num_neurons, torch.zeros((num_neurons, num_neurons)).cuda()

    def _compute_extra_head_similarity(self, w_a: Tensor, w_b: Tensor, params_name: str, p: str, sim_matrix: Tensor):
        """
        Computes similarity between extra-head weights for attention layers.
        Updates the similarity matrix in-place.
        """
        if self.normalize_weights:
            norms_a = torch.norm(w_a, dim=(1, 2))
            mean_norm_heads_a = norms_a.mean()
            norms_b = torch.norm(w_b, dim=(1, 2))
            normalized_heads_b = w_b / norms_b.view(self.num_heads, 1, 1)
            normalized_heads_b = normalized_heads_b * mean_norm_heads_a
            normalized_heads_a = (
                w_a / norms_a.view(self.num_heads, 1, 1)) * mean_norm_heads_a
            w_a = normalized_heads_a
            w_b = normalized_heads_b

        # k=head_dim
        sim_matrix += singular_values_norm_multihead(w_a, w_b, k=w_a.shape[1])

    def _compute_intra_head_similarity(self, w_a: Tensor, w_b: Tensor, p: str, sim_matrix: Dict[str, Tensor]):
        """
        Computes similarity between intra-head weights for attention layers.
        Updates the similarity matrix in-place for each head.
        """
        for i in range(self.num_heads):
            sim_matrix[f"sim_matrix_{i}"] += w_a[i] @ w_b[self.all_perm_indices[p][i]].T
        
        
    def _process_attention_layer(self, w_a: Tensor, w_b: Tensor, params_name: str, p: str, sim_matrix: Union[Tensor, Dict[str, Tensor]], intra_head_phase: bool):
        """
        Processes an attention layer, reshaping and normalizing weights, and updating similarity matrices.
        """
        if "bias" in params_name:
            return
        head_dim = w_a.shape[1] // self.num_heads
        w_a = w_a.reshape(self.num_heads, head_dim, -1)
        w_b = w_b.reshape(self.num_heads, head_dim, -1)
        try:
            bias_params_name = params_name.replace("weight", "bias")
            biases_a, biases_b = self.params_a[bias_params_name], self.params_b[bias_params_name]
            w_a = torch.cat((w_a, biases_a.reshape(self.num_heads, -1).unsqueeze(2)), dim=-1)
            w_b = torch.cat((w_b, biases_b.reshape(self.num_heads, -1).unsqueeze(2)), dim=-1)
        except:
            print("Bias not found for attention's linear projections")
            
        if not intra_head_phase:
            self._compute_extra_head_similarity(w_a, w_b, params_name, p, sim_matrix)
        elif intra_head_phase and self.intra_head:
            self._compute_intra_head_similarity(w_a, w_b, p, sim_matrix)
    
    def _process_non_attention_layer(self, w_a: Tensor, w_b: Tensor, num_neurons: int, sim_matrix: Tensor):
        """
        Processes a non-attention layer, reshaping and normalizing weights, and updating the similarity matrix.
        """
        w_a, w_b = w_a.reshape(num_neurons, -1), w_b.reshape(num_neurons, -1)
        if self.normalize_weights:
            w_a = w_a / torch.norm(w_a, dim=1, keepdim=True)
            w_b = w_b / torch.norm(w_b, dim=1, keepdim=True)
        sim_matrix += w_a @ w_b.T
    
    def _update_attention_perm_indices(self, p: str, sim_matrix: Union[Tensor, Dict[str, Tensor]], intra_head_phase: bool) -> bool:
        """
        Updates permutation indices for attention layers by solving the assignment problem.
        Returns True if the permutation improved the similarity.
        """
        if not intra_head_phase:
            perm_indices = solve_linear_assignment_problem(sim_matrix, maximize=False)
            old_sim = compute_weights_similarity(sim_matrix, self.all_perm_indices[p])
            self.all_perm_indices[p] = perm_indices
            new_sim = compute_weights_similarity(sim_matrix, perm_indices)
            return new_sim < old_sim - eps
        elif intra_head_phase and self.intra_head:
            intra_progress = []
            for i in range(self.num_heads):
                head_sim_matrix = sim_matrix[f"sim_matrix_{i}"]
                perm_indices = solve_linear_assignment_problem(head_sim_matrix, maximize=True)
                old_sim = compute_weights_similarity(head_sim_matrix, self.all_heads_indices[p][f"P_head_{i}"])
                self.all_heads_indices[p][f"P_head_{i}"] = perm_indices
                new_sim = compute_weights_similarity(head_sim_matrix, perm_indices)
                intra_progress.append(new_sim > (old_sim+ eps))
            return sum(intra_progress) > 0
        
    def _update_non_attention_perm_indices(self, p: str, sim_matrix: Tensor) -> bool:
        """
        Updates permutation indices for non-attention layers by solving the assignment problem.
        Returns True if the permutation improved the similarity.
        """
        perm_indices = solve_linear_assignment_problem(sim_matrix, maximize=True)
        old_sim = compute_weights_similarity(sim_matrix, self.all_perm_indices[p])
        self.all_perm_indices[p] = perm_indices
        new_sim = compute_weights_similarity(sim_matrix, perm_indices)
        return new_sim > (old_sim + eps)

    def run(self) -> Tuple[Dict[str, Tensor], Union[Dict[str, Dict[str, Tensor]], None]]:
        """
        Runs the weight matching algorithm for a fixed number of iterations or until convergence.
        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Dict[str, Tensor]] or None]:
                Final permutation indices and (optionally) intra-head indices.
        """
        intra_head_phase = False
        extra_head_progress = False 
        for iteration in tqdm(range(self.max_iter), desc="Weight matching"):
            progress = False
            perm_order = get_layer_iteration_order(self.layer_iteration_order, self.num_layers)
            
            for p_ix in perm_order:
                p = self.perm_names[p_ix]
                num_neurons, sim_matrix = self._initialize_similarity_matrices(p, intra_head_phase)
                params_and_axes = self.ps.perm_to_layers_and_axes[p]

                for params_name, axis in params_and_axes:
                    if "identity" in params_name:
                        continue
                    w_a, w_b = self.params_a[params_name], self.params_b[params_name]
                    perms_to_apply = self.ps.layer_and_axes_to_perm[params_name]
                    w_b = get_permuted_param(w_b, perms_to_apply, self.all_perm_indices, axis, self.num_heads, self.all_heads_indices)
                    w_a, w_b = torch.moveaxis(w_a, axis, 0), torch.moveaxis(w_b, axis, 0)
                    
                    # Preprocessing
                    if self.p_trim is not None:
                        threshold_a = torch.quantile(torch.abs(w_a), self.p_trim)
                        threshold_b = torch.quantile(torch.abs(w_b), self.p_trim)
                        mask_a, mask_b = torch.abs(w_a) >= threshold_a, torch.abs(w_b) >= threshold_b
                        w_a, w_b = torch.where(mask_a, w_a, 0.0), torch.where(mask_b, w_b, 0.0)
                    
                    # Update similarity matrices
                    if "attn" in p and self.num_heads is not None:
                        self._process_attention_layer(w_a, w_b, params_name, p, sim_matrix, intra_head_phase)
                    else:
                        self._process_non_attention_layer(w_a, w_b, num_neurons, sim_matrix)
                    
                if "attn" in p and self.num_heads is not None:
                    update = self._update_attention_perm_indices(p, sim_matrix, intra_head_phase)
                    if not intra_head_phase:
                        extra_head_progress = update
                else:
                    update = self._update_non_attention_perm_indices(p, sim_matrix)
                
                progress = progress or update

            if not progress:
                break
            if extra_head_progress:
                intra_head_phase = False
            else:
                intra_head_phase = True
                
        return self.all_perm_indices, self.all_heads_indices

class BruteforceWeightMatcher(WeightMatcher):
    """
    Weight matcher that performs brute-force head-wise matching for attention layers.
    Used for ablation or baseline comparisons.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.br_heads_indices = None
        if self.num_heads is not None and "attn" in self.perm_names[0]:
            head_dim = self.perm_sizes[self.perm_names[0]] // self.num_heads
            self.br_heads_indices = torch.zeros(self.num_heads, self.num_heads, head_dim)
            self.br_heads_indices[:] = torch.arange(head_dim)

    def _initialize_similarity_matrices(self, p: str, intra_head_phase: bool) -> Tuple[int, Union[Tensor, Dict[str, Tensor]]]:
        if "attn" in p and self.num_heads is not None:
            head_dim = self.perm_sizes[p] // self.num_heads
            sim_matrices = torch.zeros((self.num_heads, self.num_heads, head_dim, head_dim), device='cuda')
            return head_dim, sim_matrices
        else:
            return super()._initialize_similarity_matrices(p, intra_head_phase)
        
    def _process_attention_layer(self, w_a: Tensor, w_b: Tensor, params_name: str, p: str, sim_matrix: Union[Tensor, Dict[str, Tensor]], intra_head_phase: bool):
        if "bias" in params_name:
            return
        self._compute_extra_head_similarity(w_a, w_b, params_name, p, sim_matrix)

    def _compute_extra_head_similarity(self, w_a: Tensor, w_b: Tensor, params_name: str, p: str, sim_matrix: Tensor):
        head_dim = w_a.shape[1] // self.num_heads
        w_a = w_a.reshape(self.num_heads, head_dim, -1)
        w_b = w_b.reshape(self.num_heads, head_dim, -1)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                sim_matrix[i, j] += w_a[i] @ w_b[j].T  #head_dim x head_dim

    def _update_attention_perm_indices(self, p: str, sim_matrix: Tensor, intra_head_phase: bool) -> bool:
        final_sim_matrix = torch.zeros(self.num_heads, self.num_heads)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                perm_indices = solve_linear_assignment_problem(sim_matrix[i,j], maximize=True)
                self.br_heads_indices[i][j] = perm_indices
                similarity = compute_weights_similarity(sim_matrix[i,j], perm_indices)
                final_sim_matrix[i,j] = similarity
                
        perm_indices = solve_linear_assignment_problem(final_sim_matrix, maximize=True)
        old_sim = compute_weights_similarity(final_sim_matrix, self.all_perm_indices[p])
        self.all_perm_indices[p] = perm_indices
        new_sim = compute_weights_similarity(final_sim_matrix, perm_indices)
        
        for i in range(self.num_heads):
            self.all_heads_indices[p][f"P_head_{i}"] = self.br_heads_indices[i, perm_indices[i]].type(torch.int32)
        return new_sim > (old_sim + eps)
    

def compute_weights_similarity(similarity_matrix, perm_indices):
    """
    Computes the total similarity between weights given a similarity matrix and permutation indices.
    Args:
        similarity_matrix (Tensor): Similarity matrix S[i, j] = w_a[i] @ w_b[j].
        perm_indices (Tensor): Indices specifying the permutation.
    Returns:
        Tensor: The total similarity value.
    """
    """
    similarity_matrix: matrix s.t. S[i, j] = w_a[i] @ w_b[j]

    we sum over the cells identified by perm_indices, i.e. S[i, perm_indices[i]] for all i
    """

    n = len(perm_indices)

    similarity = torch.sum(similarity_matrix[torch.arange(n), perm_indices.long()])

    return similarity

def solve_linear_assignment_problem(sim_matrix: Union[torch.Tensor, np.ndarray], return_matrix=False, maximize=True):
    """
    Solves the linear assignment problem (Hungarian algorithm) for a similarity matrix.
    Args:
        sim_matrix (Tensor or np.ndarray): The similarity matrix.
        return_matrix (bool): If True, returns the permutation matrix instead of indices.
        maximize (bool): Whether to maximize or minimize the assignment.
    Returns:
        Tensor: Permutation indices or matrix.
    """
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().detach().numpy()

    ri, ci = linear_sum_assignment(sim_matrix, maximize)

    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    indices = torch.tensor(ci)
    return indices if not return_matrix else perm_indices_to_perm_matrix(indices)


def solve_linear_assignment_problem_with_threshold(sim_matrix: Union[torch.Tensor, np.ndarray], return_matrix=False, maximize=True, threshold=1):
    """
    Solves the linear assignment problem with a threshold, padding the matrix as needed.
    Args:
        sim_matrix (Tensor or np.ndarray): The similarity matrix.
        return_matrix (bool): If True, returns the permutation matrix instead of indices.
        maximize (bool): Whether to maximize or minimize the assignment.
        threshold (float): Threshold for valid assignments.
    Returns:
        Tensor: Permutation indices or matrix.
    """
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().detach().numpy()

    dim = sim_matrix.shape[0]
    constant = 1000
    min_constant = -1e9
    max_constant = 1e9
    
    if maximize:
        sim_matrix = np.where(sim_matrix < threshold, min_constant, sim_matrix)
        padded_sim_matrix = np.full((dim * 2, dim * 2), min_constant)
        padded_sim_matrix[:dim, :dim] = sim_matrix
    else:
        sim_matrix = np.where(sim_matrix > threshold, max_constant, sim_matrix)
        padded_sim_matrix = np.full((dim * 2, dim * 2), max_constant)  
        padded_sim_matrix[:dim, :dim] = sim_matrix
      
    ri, ci = linear_sum_assignment(padded_sim_matrix, maximize)
    
    if maximize:
        valid_assignments = [(r, c) for r,c in zip (ri, ci) if r < dim and c < dim and sim_matrix[r, c] > min_constant]
    else:
        valid_assignments = [(r, c) for r,c in zip (ri, ci) if r < dim and c < dim and sim_matrix[r, c] < max_constant]

    ri, ci = zip(*valid_assignments)
    
    # assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    
    indices = torch.tensor(ci)
    return indices if not return_matrix else perm_indices_to_perm_matrix(indices)
