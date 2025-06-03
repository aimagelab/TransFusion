"""
Functions for model parameter interpolation and orthogonality analysis.
Includes cosine similarity computation and parameter interpolation logic.
"""
######################################################################
# Interpolation and Orthogonality Utilities
######################################################################
import torch
from copy import deepcopy


def compute_orthogonality_measure(pretrained_state_dict, finetuned_state_dict):
    orthogonality_measures = {}
    for key in pretrained_state_dict:
        param_pretrained = pretrained_state_dict[key]
        param_finetuned = finetuned_state_dict[key]
        assert param_pretrained.shape == param_finetuned.shape, f"Mismatch nelle dimensioni per {key}"
        param_pretrained_flat = param_pretrained.view(-1)
        param_finetuned_flat = param_finetuned.view(-1)
        inner_product = torch.dot(param_pretrained_flat, param_finetuned_flat)
        norm_pretrained = torch.norm(param_pretrained_flat)
        norm_finetuned = torch.norm(param_finetuned_flat)

        if norm_pretrained.item() == 0 or norm_finetuned.item() == 0:
            cos_theta = 0.0
        else:
            # Calcola il coseno dell'angolo
            cos_theta = inner_product / (norm_pretrained * norm_finetuned)

        orthogonality_measures[key] = cos_theta.item()
    return orthogonality_measures


def adjust_gamma_based_on_orthogonality(gamma, cos_theta, beta=5.0):
    # Mappa cos_theta da [-1, 1] a [0, 1]
    cos_theta_mapped = (cos_theta + 1) / 2
    # Calcola il fattore esponenziale
    adjusted_gamma = gamma * (cos_theta_mapped ** beta)
    return adjusted_gamma


def linear_interpolation(pretrained=None, finetuned=None, gamma=0.5):
    assert pretrained is not None, "Pretrained model cannot be None"
    assert finetuned is not None, "Finetuned model cannot be None"

    pretrained.eval()
    finetuned.eval()

    model_new = deepcopy(finetuned)

    with torch.no_grad():
        pretrained_state_dict = pretrained.state_dict()
        finetuned_state_dict = finetuned.state_dict()

        # Verifica che le chiavi corrispondano
        assert pretrained_state_dict.keys() == finetuned_state_dict.keys(
        ), "Models have different parameter keys"

        model = {}
        for key in pretrained_state_dict:
            param_pretrained = pretrained_state_dict[key]
            param_finetuned = finetuned_state_dict[key]

            # Interpolazione lineare
            interpolated_param = gamma * param_finetuned + \
                (1 - gamma) * param_pretrained
            model[key] = interpolated_param

        # Carica lo state_dict interpolato
        model_new.load_state_dict(model, strict=True)

    model_new.eval()
    return model_new


def orthogonal_interpolation(pretrained=None, finetuned=None, gamma=0.5, beta=5.0):
    assert pretrained is not None, "Pretrained model cannot be None"
    assert finetuned is not None, "Finetuned model cannot be None"

    pretrained.eval()
    finetuned.eval()

    model_new = deepcopy(finetuned)

    with torch.no_grad():
        pretrained_state_dict = pretrained.state_dict()
        finetuned_state_dict = finetuned.state_dict()

        # Verifica che le chiavi corrispondano
        assert pretrained_state_dict.keys() == finetuned_state_dict.keys(
        ), "Models have different parameter keys"

        # Calcola le misure di ortogonalità
        orthogonality_measures = compute_orthogonality_measure(
            pretrained_state_dict, finetuned_state_dict)

        model = {}
        for key in pretrained_state_dict:
            cos_theta = orthogonality_measures[key]
            if ('ln_' in key):
                model[key] = pretrained_state_dict[key]
            else:
                adjusted_gamma = adjust_gamma_based_on_orthogonality(
                    gamma, cos_theta, beta)

                param_pretrained = pretrained_state_dict[key]
                param_finetuned = finetuned_state_dict[key]

                sign_pretrained = torch.sign(param_pretrained)
                sign_finetuned = torch.sign(param_finetuned)

                same_sign_mask = (sign_pretrained == sign_finetuned) & (
                    sign_pretrained != 0) & (sign_finetuned != 0)

                interpolated_param = torch.where(
                    same_sign_mask,
                    adjusted_gamma * param_finetuned +
                    (1 - adjusted_gamma) * param_pretrained,
                    param_pretrained
                )
                model[key] = interpolated_param

        # Carica lo state_dict interpolato
        model_new.load_state_dict(model, strict=True)

    model_new.eval()
    return model_new


def interpolate(pretrained=None, finetuned=None, gamma=0.5, beta=10.0, strategy='linear'):
    assert pretrained is not None, "Pretrained model cannot be None"
    assert finetuned is not None, "Finetuned model cannot be None"

    if strategy == 'linear':
        return linear_interpolation(pretrained, finetuned, gamma)
    elif strategy == 'orthogonal':
        return orthogonal_interpolation(pretrained, finetuned, gamma, beta)
    else:
        raise ValueError(f"Invalid interpolation strategy: {strategy}")
