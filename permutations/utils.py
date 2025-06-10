"""
Permutation utilities for weight matching and analysis in deep learning models.

Main functionalities:
- Conversion between permutation indices and matrices
- Application of permutations to tensors and model parameters
- Validation and inversion of permutations
- Similarity metrics for comparing model weights or attention heads
- Utilities for loading, unfactoring, and applying permutations to state_dicts and task vectors
"""
######################################################################
# Permutation Utilities
######################################################################
import torch.nn.functional as F
import copy
import itertools
import json
import logging
from typing import Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from torch import Tensor

from .permutation_spec import PermutationSpec
from torch.nn.functional import cosine_similarity

from scipy.linalg import orthogonal_procrustes
from scipy.stats import wasserstein_distance
# shape (n, n), contains the permutation matrix, e.g. [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
PermutationMatrix = Tensor

# shape (n), contains the indices of the target permutation, e.g. [0, 3, 2, 1]
PermutationIndices = Tensor

pylogger = logging.getLogger(__name__)


def get_all_symbols_combinations(symbols: Set[str]) -> List[Tuple[str, str]]:
    """
    Given a set of symbols, returns all possible ordered pairs (permutations of two symbols).
    Used for generating all possible relationships between permutation domains.
    Args:
        symbols: set of str, e.g. {"a", "b", "c"}
    Returns:
        list of tuple(str, str), e.g. [("a", "b"), ("a", "c"), ...]
    """
    combinations = list(itertools.permutations(symbols, 2))
    sorted_combinations = sorted(combinations)
    return sorted_combinations


def get_inverse_permutations(permutations: Dict[str, PermutationIndices]) -> Dict[str, PermutationIndices]:
    """
    Given a dictionary of permutations, returns a dictionary of the inverse permutations.
    Each permutation can be given as indices or as a matrix.
    Returns the inverse in the same format.
    """

    inv_permutations = {}

    for perm_name, perm in permutations.items():
        if perm.dim() == 1:
            perm_matrix = perm_indices_to_perm_matrix(perm)
        else:
            perm_matrix = perm

        inv_perm_matrix = perm_matrix.T

        if perm.dim() == 1:
            inv_permutations[perm_name] = perm_matrix_to_perm_indices(
                inv_perm_matrix)
        else:
            inv_permutations[perm_name] = inv_perm_matrix

    return inv_permutations


def perm_indices_to_perm_matrix(perm_indices: PermutationIndices):
    """
    Convert permutation indices to a permutation matrix.
    Args:
        perm_indices: shape (n,), e.g. [0, 2, 1]
    Returns:
        perm_matrix: shape (n, n), one-hot rows
    """
    n = len(perm_indices)
    perm_matrix = torch.eye(n, device=perm_indices.device)[perm_indices.long()]
    return perm_matrix


def perm_matrix_to_perm_indices(perm_matrix: PermutationMatrix):
    """
    Convert a permutation matrix to permutation indices.
    Args:
        perm_matrix: shape (n, n)
    Returns:
        perm_indices: shape (n,)
    """
    return perm_matrix.nonzero()[:, 1].long()


def check_permutations_are_valid(permutation, inv_permutation):
    """
    Check that each permutation and its inverse are valid and mutually inverse.
    Used for debugging and validation of permutation logic.
    """
    for layer_perm, layer_perm_inv in zip(permutation.values(), inv_permutation.values()):
        perm_matrix = perm_indices_to_perm_matrix(layer_perm)
        inv_perm_matrix = perm_indices_to_perm_matrix(layer_perm_inv).T
        assert is_valid_permutation_matrix(perm_matrix)
        assert is_valid_permutation_matrix(inv_perm_matrix)
        assert torch.all(perm_matrix == inv_perm_matrix)


def is_valid_permutation_matrix(matrix):
    """
    Check if a matrix is a valid permutation matrix (square, binary, one per row/col).
    Returns True if valid, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    row_sums = torch.sum(matrix, dim=1)
    col_sums = torch.sum(matrix, dim=0)
    ones_tensor = torch.ones_like(row_sums)
    return (
        torch.all(row_sums == ones_tensor)
        and torch.all(col_sums == ones_tensor)
        and torch.all((matrix == 0) | (matrix == 1))
    )


def perm_rows(x, perm):
    """
    Permute the first axis (rows) of tensor x by permutation matrix perm.
    Used for applying permutations to model weights (e.g., linear/conv layers).
    Args:
        x: tensor, shape (n, ...)
        perm: permutation matrix, shape (n, n)
    Returns:
        permuted x
    """
    assert x.shape[0] == perm.shape[0]
    assert perm.dim() == 2 and perm.shape[0] == perm.shape[1]

    input_dims = "jklm"[: x.dim()]
    output_dims = "iklm"[: x.dim()]

    ein_string = f"ij,{input_dims}->{output_dims}"

    return torch.einsum(ein_string, perm, x)


def perm_cols(x, perm):
    """
    Permute the second axis (columns) of tensor x by permutation matrix perm.
    Used for applying permutations to model weights (e.g., linear/conv layers).
    Args:
        x: tensor, shape (..., n)
        perm: permutation matrix, shape (n, n)
    Returns:
        permuted x
    """
    assert x.shape[1] == perm.shape[0]
    assert perm.shape[0] == perm.shape[1]

    x = x.transpose(1, 0)
    perm = perm.transpose(1, 0)

    permuted_x = perm_rows(x=x, perm=perm)

    return permuted_x.transpose(1, 0)


def get_permuted_param(param, perms_to_apply, perm_matrices, except_axis=None, num_heads=12, all_heads_indices=None):
    """
    Apply all relevant permutations to a parameter tensor, as specified by perms_to_apply.
    Handles both standard and multi-head (attention) permutations.
    Args:
        param: tensor to permute
        perms_to_apply: list of permutation names (or None)
        perm_matrices: dict of permutation matrices/indices
        except_axis: axis to skip (optional)
        num_heads: number of attention heads (if relevant)
        all_heads_indices: dict of per-head permutations (if relevant)
    Returns:
        permuted param
    """

    for axis, perm_id in enumerate(perms_to_apply):

        if axis == except_axis or perm_id is None:
            continue

        perm = perm_matrices[perm_id].cuda()
        if perm.dim() == 1:
            if 'attn' in perm_id:

                original_shape = param.shape
                if all_heads_indices is None:
                    if axis == 0:
                        param = param.reshape(num_heads, -1)
                        # opt 1 is the same as opt2
                        param = torch.index_select(param, axis, perm.int())
                    elif axis == 1:
                        # Anti-permutation of heads
                        param = param.T
                        param = torch.index_select(param.reshape(
                            num_heads, param.shape[0]//num_heads, param.shape[1]), 0, perm)
                        # param = torch.index_select(param.reshape(num_heads, -1), 0, perm)
                        param = param.reshape(
                            original_shape[1], original_shape[0])
                        param = param.T
                    param = param.reshape(original_shape)
                else:
                    heads_perm = all_heads_indices[perm_id]
                    if axis == 0:
                        if len(param.shape) > 1:
                            param = param.reshape(
                                num_heads, original_shape[0]//num_heads, -1)
                            # opt 1 is the same as opt2
                            param = torch.index_select(param, axis, perm.int())
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i], axis, heads_perm[f'P_head_{i}'].cuda())
                            param = param.transpose(1, 0)
                        else:
                            param = param.reshape(num_heads, -1)
                            # opt 1 is the same as opt2
                            param = torch.index_select(param, axis, perm.int())
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i], axis, heads_perm[f'P_head_{i}'].cuda())
                            param = param.transpose(1, 0)

                    elif axis == 1:
                        if len(param.shape) > 1:
                            param = param.T
                            param = param.reshape(
                                num_heads, param.shape[0]//num_heads, param.shape[1])
                            param = torch.index_select(param, 0, perm)
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i], 0, heads_perm[f'P_head_{i}'].cuda())
                            param = param.transpose(1, 0)
                            param = param.reshape(
                                original_shape[1], original_shape[0])
                            param = param.T
                        else:
                            param = param.T
                            param = param.reshape(num_heads, -1)
                            param = torch.index_select(param, 0, perm)
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i], 0, heads_perm[f'P_head_{i}'].cuda())
                            param = param.transpose(1, 0)
                            param = param.reshape(
                                original_shape[1], original_shape[0])
                            param = param.T

                    param = param.reshape(original_shape)

            else:
                # permute by indices
                param = torch.index_select(param, axis, perm.int())
        else:
            # permute by matrix
            param = perm_tensor_by_perm_matrix(param, perm, axis)


    return param


def cosine_similarity_models(model1_dict, model2_dict):
    """
    Compute the mean cosine similarity between all matching parameters in two model state_dicts.
    Skips batch norm tracking and identity layers.
    """
    similarities = []
    for key in model1_dict:
        if 'num_batches_tracked' in key or 'identity' in key:
            continue
        if key in model2_dict:
            similarities.append(
                cosine_similarity(model1_dict[key].flatten(
                ), model2_dict[key].flatten(), dim=0)
            )
    return sum(similarities) / len(similarities)


def svd_similarity(model1_dict, model2_dict):
    """
    Compute the mean difference of singular values between all matching parameters in two model state_dicts.
    Only compares parameters with at least 2 dimensions.
    """
    svd_diffs = []

    for key in model1_dict:
        if key in model2_dict:
            weight1 = model1_dict[key]
            weight2 = model2_dict[key]

            # Controlla che il tensore abbia almeno 2 dimensioni
            if weight1.dim() > 1 and weight2.dim() > 1:
                # Reshape dei pesi in una matrice 2D se necessario
                weight1_flat = weight1.view(weight1.size(0), -1)
                weight2_flat = weight2.view(weight2.size(0), -1)

                # Applica SVD ai pesi
                u1, s1, v1 = torch.svd(weight1_flat)
                u2, s2, v2 = torch.svd(weight2_flat)

                # Calcola la differenza tra i valori singolari
                svd_diff = torch.norm(s1 - s2).item()
                svd_diffs.append(svd_diff)
            else:
                # Gestisci il caso dei tensori con una sola dimensione o scalari
                svd_diffs.append(0.0)  # O qualsiasi altra metrica adeguata

    # Restituisce la differenza media tra i valori singolari
    return sum(svd_diffs) / len(svd_diffs)


def l2_norm_models(model1, model2):
    """
    Calculate the L2 norm of the difference between two state dictionaries.
    Skips batch norm tracking and identity layers.
    """
    """Calculate the L2 norm of the difference between two state dictionaries."""
    sum_diff_squared = 0
    for key in model2.keys():
        if "num_batches_tracked" in key or 'identity' in key:
            continue
        diff_squared_sum = torch.sum((model1[key] - model2[key]) ** 2)
        sum_diff_squared += diff_squared_sum
    return torch.sqrt(sum_diff_squared)


def perm_tensor_by_perm_matrix(tens, perm, axis):
    """
    Permute a tensor along the specified axis using a permutation matrix.
    Used for applying permutations to model weights.
    Args:
        tens: tensor to permute
        perm: permutation matrix
        axis: 0 (rows) or 1 (columns)
    Returns:
        permuted tensor
    """
    assert axis == 0 or axis == 1
    if axis == 0:
        tens = perm_rows(tens, perm)
    else:
        tens = perm_cols(tens, perm.T)

    return tens


def apply_permutation_to_statedict(ps: PermutationSpec, perm_matrices, model_a_dict, model_b_dict=None, heads_permutation=None, skip_params=False, num_heads=12):
    """
    Apply a set of permutations to a model's state_dict, according to a PermutationSpec.
    Optionally supports skipping parameters not in the spec, and per-head permutations for attention.
    Returns a new permuted state_dict.
    """
    """Apply a `perm` to `params`."""

    permuted_params = {}

    for param_name, param in model_a_dict.items():

        param_name_in_perm_dict = param_name

        # if ("num_batches_tracked" in param_name
        #     or "to_patch_tokens.1" in param_name
        #     or "temperature" in param_name):

        #     permuted_params[param_name] = param
        #     continue

        # if model_b_dict is not None:
        #     if "visual.attnpool" in param_name:
        #         permuted_params[param_name] = model_b_dict[param_name]
        #         continue
        if skip_params:
            if param_name_in_perm_dict not in ps.layer_and_axes_to_perm:
                permuted_params[param_name] = param
                continue
        else:
            assert (
                param_name_in_perm_dict in ps.layer_and_axes_to_perm
            ), f"param_name {param_name} not found in ps.layer_and_axes_to_perm"

        try:
            param = copy.deepcopy(param)
            perms_to_apply = ps.layer_and_axes_to_perm[param_name_in_perm_dict]
            # if param_name == "visual.attnpool.k_proj.weight":
            #     perm_matrices["P_attn"] = perm_matrix_to_perm_indices(perm_indices_to_perm_matrix(perm_matrices["P_attn"]).T)
            #     param = get_permuted_param(param, perms_to_apply, perm_matrices)
            #     perm_matrices["P_attn"] = perm_matrix_to_perm_indices(perm_indices_to_perm_matrix(perm_matrices["P_attn"]).T)

            param = get_permuted_param(
                param, perms_to_apply, perm_matrices, all_heads_indices=heads_permutation, num_heads=num_heads)
            permuted_params[param_name] = param
        except:
            print(
                f"Problem during application of permutation {perms_to_apply} on layer {param_name}")

    return permuted_params


def apply_permutation_to_task_vector(ps: PermutationSpec, perm_matrices, model_a_dict, model_b_dict, task_vector_dict, heads_permutation=None, skip_params=False, num_heads=12):
    """
    Apply a set of permutations to a task vector (difference of state_dicts), according to a PermutationSpec.
    Optionally supports skipping parameters not in the spec, and per-head permutations for attention.
    Returns a new permuted task vector.
    """
    """Apply a `perm` to `params`."""

    permuted_params = {}
    linear_params = [param for param in model_a_dict.keys()
                     if "fc" in param or param == 'proj']

    for param_name, param in task_vector_dict.items():

        param_name_in_perm_dict = param_name

        if skip_params:
            if param_name_in_perm_dict not in ps.layer_and_axes_to_perm:
                permuted_params[param_name] = param
                continue
        else:
            assert (
                param_name_in_perm_dict in ps.layer_and_axes_to_perm
            ), f"param_name {param_name} not found in ps.layer_and_axes_to_perm"

        try:
            param = copy.deepcopy(param)
            perms_to_apply = ps.layer_and_axes_to_perm[param_name_in_perm_dict]
            # if any(param_name in item for item in linear_params):
            if 'bias' in param_name or 'ln_' in param_name:
                permuted_params[param_name] = torch.zeros_like(param)
            else:
                threshold = 0
                param = get_permuted_param(
                    param, perms_to_apply, perm_matrices, all_heads_indices=heads_permutation, num_heads=num_heads)

                model_a_param = get_permuted_param(
                    model_a_dict[param_name], perms_to_apply, perm_matrices, all_heads_indices=heads_permutation, num_heads=num_heads)

                model_b_param = model_b_dict[param_name]

                cosine_sim_matrix = (model_b_param @ model_a_param.T)/(torch.norm(
                    model_b_param, dim=1).unsqueeze(1) * torch.norm(model_a_param, dim=1).unsqueeze(0))
                cosine_sim_vector = torch.diag(cosine_sim_matrix)

                mask = cosine_sim_vector < threshold
                print(
                    f"Mantained params for {param_name}: {mask.sum()} of {len(mask)}")
                permuted_params[param_name] = param * mask.unsqueeze(1)

                # try:
                #     bias_params_name = param_name.replace("weight", "bias")
                #     permuted_params[bias_params_name] = task_vector_dict[bias_params_name] * mask
                # except:
                #     print("Bias not found for param_name")
            # else:
            #     param = get_permuted_param(param, perms_to_apply, perm_matrices, all_heads_indices=heads_permutation, num_heads=num_heads)
            #     permuted_params[param_name] = param

        except Exception as e:
            print(
                f"{e} during application of permutation {perms_to_apply} on layer {param_name}")
            permuted_params[param_name] = permuted_params[param_name] = torch.zeros_like(
                param)

    return permuted_params


def unfactor_permutations(permutations, matrix_format=False):
    """
    Given a factored dictionary of permutations, compute all pairwise composed permutations.
    Used for reconstructing full permutation relationships from factored representations.
    """
    if matrix_format:
        raise NotImplementedError

    symbols = set(permutations.keys())

    unfactored_permutations = {
        symbol: {
            permutee: {perm: None for perm in permutations[symbol].keys()} for permutee in symbols.difference(symbol)
        }
        for symbol in symbols
    }
    for symbol, perms in permutations.items():
        for perm_name, perm in perms.items():
            if perm is not None:
                permutations[symbol][perm_name] = torch.tensor(perm)

    combinations = get_all_symbols_combinations(symbols)
    for fixed, permutee in combinations:
        for perm in permutations[fixed].keys():
            res = (
                perm_indices_to_perm_matrix(permutations[fixed][perm])
                @ perm_indices_to_perm_matrix(permutations[permutee][perm]).T
            )

            unfactored_permutations[fixed][permutee][perm] = perm_matrix_to_perm_indices(
                res)

    return unfactored_permutations


def load_permutations(path, factored=False, matrix_format=False) -> Dict[str, Union[PermutationIndices, PermutationMatrix]]:
    """
    Load permutations from a JSON file, optionally unfactoring or converting to matrix format.
    Returns a nested dictionary of permutations.
    """
    with open(path, "r") as f:
        permutations = json.load(f)

    if factored:
        return unfactor_permutations(permutations, matrix_format)

    if matrix_format:
        for source, targets in permutations.items():
            for target, source_target_perms in targets.items():
                for perm_name, perm in source_target_perms.items():
                    if perm is not None:
                        permutations[source][target][perm_name] = torch.tensor(
                            perm)

        return permutations
    else:
        for source, targets in permutations.items():
            for target, source_target_perms in targets.items():
                for perm_name, perm in source_target_perms.items():
                    if perm is not None:
                        permutations[source][target][perm_name] = torch.tensor(
                            perm)

        return permutations

def permute_batchnorm(model, perm, perm_dict, map_param_index):
    """
    Apply a permutation to the running_mean and running_var of BatchNorm layers in a model.
    Used to keep batch norm statistics consistent after permuting weights.
    """

    for name, module in model.named_modules():

        if "BatchNorm" in str(type(module)):

            if name + ".weight" in map_param_index:

                if module.running_mean is None and module.running_var is None:
                    continue

                i = perm_dict[map_param_index[name + ".weight"]]

                index = torch.argmax(perm[i], dim=1) if i is not None else torch.arange(
                    module.running_mean.shape[0])

                module.running_mean.copy_(module.running_mean[index, ...])
                module.running_var.copy_(module.running_var[index, ...])


def lerp(t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Linear interpolation between two vectors or arrays.
    Used for vector arithmetic, not specific to permutations.
    """
    return (1 - t) * v0 + t * v1


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    if not isinstance(v0, np.ndarray):
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = v0 / (np.linalg.norm(v0) + 1e-6)
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return res

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res


def generalized_inner_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the generalized inner product between two tensors A and B using einsum.
    Used for advanced similarity or alignment metrics.
    """
    """
    Compute the generalized inner product between two tensors A and B.

    Args:
        A: tensor
        B: tensor

    Returns:
        result: tensor

    """

    A_dims = "ijkm"[0: A.dim()]
    B_dims = "jnkm"[0: B.dim()]

    result = torch.einsum(f"{A_dims}, {B_dims} -> in", A, B)

    return result


def singular_values_norm_multihead(a: torch.Tensor, b: torch.Tensor, k=10) -> torch.Tensor:
    """
    Compute the norm of the difference between the top-k singular values of each head in a and b.
    Used for comparing multi-head attention weights.
    """
    svd_vals_a = torch.linalg.svdvals(a)[:, :k]
    svd_vals_b = torch.linalg.svdvals(b)[:, :k]
    # differenze tra vettori di valori singolari all head_a vs all_head_b
    diff_sings = svd_vals_a.unsqueeze(1) - svd_vals_b.unsqueeze(0)
    res = torch.norm(diff_sings, dim=-1)
    # _, res_idx =  linear_sum_assignment(res)
    # return res, res_idx
    return res


def singular_values_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the norm of the difference between the singular values of a and b.
    Used for comparing matrices (e.g., weights).
    """
    svd_vals_a = torch.linalg.svdvals(a)
    svd_vals_b = torch.linalg.svdvals(b)
    # differenze tra vettori di valori singolari all head_a vs all_head_b
    diff_sings = svd_vals_a.unsqueeze(1) - svd_vals_b.unsqueeze(0)
    res = torch.norm(diff_sings, dim=-1)
    # _, res_idx =  linear_sum_assignment(res)
    # return res, res_idx
    return res


def gram_matrix(a, b):
    """
    Compute the Gram matrix (dot products) between all pairs of heads in a and b.
    Used for attention head similarity.
    """
    return torch.einsum('ihd,jhd->ij', a, b)


def measure(a, b, k=10):
    """
    Compute a norm-based similarity between all pairs of heads in a and b.
    Used for attention head similarity.
    """
    res = (a.unsqueeze(1)@b.unsqueeze(0).transpose(-2, -1))
    res = torch.norm(res, dim=(-1, -2))
    return res


def singular_value_similarity_matrix(a, b, k=10):
    """
    Compute a similarity matrix between heads in a and b using Wasserstein distance on top-k singular values.
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    # Compute singular values for each head in A and B
    svd_vals_a = torch.stack([torch.linalg.svdvals(a[i])[:k]
                             for i in range(num_heads_a)])
    svd_vals_b = torch.stack([torch.linalg.svdvals(b[j])[:k]
                             for j in range(num_heads_b)])

    # Initialize similarity matrix
    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    # Compute distance between each head in A and each head in B
    for i in range(num_heads_a):
        for j in range(num_heads_b):
            # Compute Wasserstein distance (Earth Mover's Distance) between singular values
            similarity_matrix[i, j] = wasserstein_distance(
                svd_vals_a[i].cpu().numpy(), svd_vals_b[j].cpu().numpy())

    return similarity_matrix


def procrustes_similarity_matrix(a, b):
    """
    Compute a similarity matrix between heads in a and b using Procrustes alignment (Frobenius norm after optimal rotation).
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    # Initialize similarity matrix
    similarity_matrix = np.zeros((num_heads_a, num_heads_b))

    # Compute Procrustes distance for each head in A against each head in B
    for i in range(num_heads_a):
        for j in range(num_heads_b):
            matrix_a = a[i].cpu().numpy()
            matrix_b = b[j].cpu().numpy()

            # Perform the orthogonal Procrustes alignment
            R, scale = orthogonal_procrustes(matrix_a, matrix_b)
            aligned_a = matrix_a @ R

            # Calculate the Frobenius norm of the difference
            similarity_matrix[i, j] = np.linalg.norm(
                aligned_a - matrix_b, 'fro')

    return similarity_matrix


def covariance_similarity_matrix(a, b):
    """
    Compute a similarity matrix between heads in a and b using Frobenius norm of covariance matrices.
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    # Initialize similarity matrix
    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    # Compute covariance distance for each head in A against each head in B
    for i in range(num_heads_a):
        cov_a = torch.cov(a[i].T)  # Covariance matrix of head i in A
        for j in range(num_heads_b):
            cov_b = torch.cov(b[j].T)  # Covariance matrix of head j in B

            # Calculate the Frobenius norm between the covariance matrices
            similarity_matrix[i, j] = torch.norm(cov_a - cov_b, p='fro')

    return similarity_matrix


def spectral_similarity_matrix(a, b, k=10):
    """
    Compute a similarity matrix between heads in a and b using Wasserstein distance on top-k eigenvalues of adjacency matrices.
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    for i in range(num_heads_a):
        # Create the adjacency matrix for head i in A
        adj_a = a[i] @ a[i].T
        eigenvals_a, _ = torch.linalg.eigh(adj_a)
        eigenvals_a = eigenvals_a[-k:]  # Take the top k eigenvalues

        for j in range(num_heads_b):
            # Create the adjacency matrix for head j in B
            adj_b = b[j] @ b[j].T
            eigenvals_b, _ = torch.linalg.eigh(adj_b)
            eigenvals_b = eigenvals_b[-k:]  # Take the top k eigenvalues

            # Compute Wasserstein distance between eigenvalues
            similarity_matrix[i, j] = wasserstein_distance(
                eigenvals_a.cpu().numpy(), eigenvals_b.cpu().numpy())

    return similarity_matrix


def bures_wasserstein_similarity_matrix(a, b):
    """
    Compute a similarity matrix between heads in a and b using the Bures-Wasserstein distance between covariance matrices.
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    for i in range(num_heads_a):
        cov_a = torch.cov(a[i].T)
        for j in range(num_heads_b):
            cov_b = torch.cov(b[j].T)

            # Calculate matrix square root of cov_a
            # Adding a small epsilon for stability
            sqrt_cov_a = torch.linalg.cholesky(
                cov_a + 1e-6 * torch.eye(cov_a.size(0)))

            # Compute the intermediate term for Bures-Wasserstein distance
            product = sqrt_cov_a @ cov_b @ sqrt_cov_a
            # Square root of the product matrix
            sqrt_product = torch.linalg.cholesky(
                product + 1e-6 * torch.eye(product.size(0)))

            # Compute the Bures-Wasserstein distance
            similarity_matrix[i, j] = torch.trace(
                cov_a + cov_b - 2 * sqrt_product)

    return similarity_matrix


def cosine_similarity_matrix(a, b):
    """
    Compute a similarity matrix between heads in a and b using cosine similarity of flattened weights.
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]

    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    for i in range(num_heads_a):
        for j in range(num_heads_b):
            head_a_flat = a[i].flatten()
            head_b_flat = b[j].flatten()

            # Calculate cosine similarity
            similarity_matrix[i, j] = F.cosine_similarity(
                head_a_flat, head_b_flat, dim=0)

    return similarity_matrix


def cca_similarity_matrix(a, b, num_components=5):
    """
    Compute a similarity matrix between heads in a and b using Canonical Correlation Analysis (CCA).
    Used for matching attention heads.
    """
    num_heads_a = a.shape[0]
    num_heads_b = b.shape[0]
    similarity_matrix = torch.zeros((num_heads_a, num_heads_b))

    for i in range(num_heads_a):
        for j in range(num_heads_b):
            head_a = a[i]
            head_b = b[j]

            # Center the matrices
            head_a = head_a - head_a.mean(dim=0)
            head_b = head_b - head_b.mean(dim=0)

            # Perform SVD for CCA on the centered matrices
            u_a, s_a, _ = torch.svd(head_a)
            u_b, s_b, _ = torch.svd(head_b)

            # Select the top components based on `num_components`
            top_u_a = u_a[:, :num_components]
            top_u_b = u_b[:, :num_components]

            # Compute correlation distance as 1 - mean correlation
            correlations = [torch.corrcoef(torch.stack([top_u_a[:, k], top_u_b[:, k]]))[
                0, 1] for k in range(num_components)]
            similarity_matrix[i, j] = 1 - torch.mean(torch.stack(correlations))

    return similarity_matrix


def cca(activation1, activation2):
    """
    Compute Canonical Correlation Analysis (CCA) between two activation matrices.
    Returns the canonical correlations for each pair.
    Used for comparing representations.
    """
    """
    Compute Canonical Correlation Analysis (CCA) between two activation matrices.

    Args:
        activation1 (torch.Tensor): Activation matrix from model 1, shape (n_samples, n_features1).
        activation2 (torch.Tensor): Activation matrix from model 2, shape (n_samples, n_features2).

    Returns:
        torch.Tensor: Canonical correlations for each canonical pair.
    """
    # Center the activations
    activation1 = activation1 - activation1.mean(dim=0, keepdim=True)
    activation2 = activation2 - activation2.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    covariance_11 = torch.matmul(
        activation1.T, activation1) / (activation1.shape[0] - 1)
    covariance_22 = torch.matmul(
        activation2.T, activation2) / (activation2.shape[0] - 1)
    covariance_12 = torch.matmul(
        activation1.T, activation2) / (activation1.shape[0] - 1)

    # Compute inverses of covariance matrices
    inv_cov_11 = torch.linalg.pinv(covariance_11)
    inv_cov_22 = torch.linalg.pinv(covariance_22)

    # Compute cross-covariance terms
    mat_a = torch.matmul(torch.matmul(inv_cov_11, covariance_12),
                         torch.matmul(inv_cov_22, covariance_12.T))
    mat_b = torch.matmul(torch.matmul(inv_cov_22, covariance_12.T),
                         torch.matmul(inv_cov_11, covariance_12))

    # Eigenvalue decomposition
    eigvals_a, _ = torch.linalg.eig(mat_a)
    eigvals_b, _ = torch.linalg.eig(mat_b)

    # Canonical correlations are the square roots of the real parts of the eigenvalues
    canonical_correlations = torch.sqrt(torch.real(eigvals_a))

    return canonical_correlations


def earth_movers_distance(p, q):
    """
    Compute the Earth Mover's Distance (Wasserstein Distance) between two 1D distributions.
    Used for comparing distributions of singular values or eigenvalues.
    """
    """
    Compute the Earth Mover's Distance (Wasserstein Distance) between two 1D distributions.
    """
    p_sorted, _ = torch.sort(p, dim=-1)
    q_sorted, _ = torch.sort(q, dim=-1)
    cdf_p = torch.cumsum(p_sorted, dim=-1)
    cdf_q = torch.cumsum(q_sorted, dim=-1)
    cdf_diff = torch.abs(cdf_p - cdf_q)
    emd = torch.sum(cdf_diff, dim=-1) / p.shape[-1]
    return emd


def batched_emd_similarity_matrix(A, B, batch_size=100):
    """
    Compute the EMD similarity matrix in batches to save memory.
    Used for large-scale comparison of distributions.
    """
    """
    Compute the EMD similarity matrix in batches to save memory.

    Args:
        A (torch.Tensor): A 2D tensor of shape (A_rows, features).
        B (torch.Tensor): A 2D tensor of shape (B_rows, features).
        batch_size (int): Number of rows to process at a time.

    Returns:
        torch.Tensor: A similarity matrix of shape (A_rows, B_rows).
    """
    A_rows, features = A.shape
    B_rows, _ = B.shape
    similarity_matrix = torch.zeros((A_rows, B_rows), device=A.device)

    # Process A in batches
    for i in range(0, A_rows, batch_size):
        A_batch = A[i:i+batch_size]  # Shape: (batch_size, features)

        # Process B in batches
        for j in range(0, B_rows, batch_size):
            B_batch = B[j:j+batch_size]  # Shape: (batch_size, features)

            # Compute similarity for the current batch pair
            # Shape: (batch_size, 1, features)
            A_expanded = A_batch.unsqueeze(1)
            # Shape: (1, batch_size, features)
            B_expanded = B_batch.unsqueeze(0)
            # Shape: (batch_size, batch_size)
            emd_batch = earth_movers_distance(A_expanded, B_expanded)
            similarity_batch = 1 / (1 + emd_batch)

            # Store the result in the full similarity matrix
            similarity_matrix[i:i+batch_size,
                              j:j+batch_size] = similarity_batch

    return similarity_matrix
