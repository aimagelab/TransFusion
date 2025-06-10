import logging
logger = logging.getLogger(__name__)
"""
Implements the TaskVector class for model difference arithmetic.
Supports initialization from checkpoints or vectors and arithmetic operations.
"""
######################################################################
# Task Vector Class
######################################################################
import numbers
import torch
import math
from copy import deepcopy


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                try:
                    pretrained_state_dict = torch.load(pretrained_checkpoint)[
                        'model_state_dict']
                except AttributeError:
                    pretrained_state_dict = pretrained_checkpoint.state_dict()
                # pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                # pretrained_state_dict = torch.load(pretrained_checkpoint)['model_state_dict']
                try:
                    finetuned_state_dict = torch.load(finetuned_checkpoint)[
                        'model_state_dict']
                except AttributeError:
                    finetuned_state_dict = finetuned_checkpoint.state_dict()
                # finetuned_state_dict = torch.load(finetuned_checkpoint)['model_state_dict']
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        logger.info(f"Skipping key (int/uint): {key}")
                        continue
                    self.vector[key] = finetuned_state_dict[key] - \
                        pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    logger.warning(f'Key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def __mul__(self, alpha):
        if isinstance(alpha, (int, float)):
            new_vector = {}
            for key in self.vector:
                new_vector[key] = alpha * self.vector[key]
            return TaskVector(vector=new_vector)
        elif isinstance(alpha, TaskVector):
            new_vector = {}
            for k1, k2 in zip(self.vector.keys(), alpha.vector.keys()):
                new_vector[k1] = self.vector[k1] * self.alpha[k2]
            return TaskVector(vector=new_vector)

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def __sub__(self, other):
        new_vector = {}
        for key in self.vector.keys():
            new_vector[key] = self.vector[key] - other.vector[key]
        return TaskVector(vector=new_vector)

    def svd(self, percentage, q=50):
        new_vector = {}
        for key in self.vector.keys():
            if len(self.vector[key].shape) != 2:
                new_vector[key] = self.vector[key]
            # elif len(self.vector[key].shape) > 2:
            #     reduced_filters = []
            #     for i in range(self.vector[key].shape[0]):
            #         filter_2d = self.vector[key].view(self.vector[key].shape[0], -1)
            #         U, S, Vh = torch.linalg.svd(filter_2d)
            #         r = int(percentage * len(S))
            #         U_r = U[:, :r]
            #         S_r = S[:r]
            #         Vh_r = Vh[:r, :]
            #         S_r_diag = torch.diag(S_r)
            #         filter_2d_reduced = U_r @ S_r_diag @ Vh_r
            #         filter_3d_reduced = filter_2d_reduced.view(self.vector[key].shape[0:])
            #         reduced_filters.append(filter_3d_reduced)
            #     new_vector[key] = torch.stack(reduced_filters)
            else:
                U, S, Vh = torch.svd_lowrank(self.vector[key], q=q, niter=6)
                r = int(percentage * len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vh = Vh.T
                Vh_r = Vh[:r, :]
                S_r_diag = torch.diag(S_r)
                new_vector[key] = U_r @ S_r_diag @ Vh_r
                new_vector[key] = new_vector[key].reshape(
                    self.vector[key].shape)
        return TaskVector(vector=new_vector)

    def apply_orthogonal_components(self, model_parameters):
        new_vector = {}
        for key in self.vector.keys():
            a = self.vector[key].reshape(-1)  # Task vector component
            b = model_parameters[key].reshape(-1)  # Model parameter component

            # Ensure data types are consistent
            if a.dtype != b.dtype:
                b = b.type(a.dtype)

            # Compute the projection of b onto a
            dot_prod = torch.dot(a, b)
            norm_a_squared = torch.dot(a, a)
            if norm_a_squared > 0:  # Avoid division by zero
                projection = (dot_prod / norm_a_squared) * a
            else:
                projection = torch.zeros_like(a)

            # Compute the orthogonal component of b with respect to a
            b_orthogonal = b - projection

            # Retain only the orthogonal component
            new_vector[key] = b_orthogonal.reshape(self.vector[key].shape)

        return TaskVector(vector=new_vector)

    def get_orthogonal_components(self, other):
        new_vector = {}
        for key in self.vector.keys():
            a = self.vector[key].view(-1)
            b = other.vector[key].view(-1)
            if a.dtype != b.dtype:
                b = b.type(a.dtype)
            dot_prod = torch.dot(a, b)
            dot_prod /= torch.dot(a, a)
            pr = dot_prod * a
            b_orthogonal = b - pr
            new_vector[key] = b_orthogonal.reshape(self.vector[key].shape)
        return TaskVector(vector=new_vector)

    def interpolate_params(self, original_params, key, alpha, method='spherical'):
        if method == 'linear':
            return original_params + (self.vector[key] * alpha)
        elif method == 'spherical':
            return None

    def get_new_params(self, dict_model, key, model_new, model_old, scaling_coef=1.0, p=0.99, beta=1.0, method='linear'):
        magnitudes = torch.abs(model_new.state_dict()[key])
        threshold = torch.quantile(magnitudes, p)
        high_magnitude_mask = magnitudes >= threshold

        ratio = torch.exp(torch.abs(model_new.state_dict()[
                          key] - model_old.state_dict()[key])*beta)
        new_params = torch.where(
            high_magnitude_mask,
            self.interpolate_params(
                dict_model[key], key, scaling_coef/ratio, method=method),
            dict_model[key])
        return new_params

    def normalize_vector(self, model_a_state_dict, model_b_state_dict):

        for key in self.vector.keys():
            norm_A = torch.norm(model_a_state_dict[key])
            norm_B = torch.norm(model_b_state_dict[key])

            # Avoid division by zero
            if norm_A > 0 and norm_B > 0:
                scale_factor = norm_B / norm_A
                # logger.info(f"Scale factor for {key}: {scale_factor}")
            else:
                scale_factor = 1.0

            self.vector[key] = scale_factor * self.vector[key]

    def proportional_normalize_vector(self, model_a_state_dict, model_b_state_dict):
        for key in self.vector.keys():
            norm_A = torch.norm(model_a_state_dict[key])
            norm_B = torch.norm(model_b_state_dict[key])
            norm_ta = torch.norm(self.vector[key])
            # Avoid division by zero
            if norm_A > 0:
                scale_factor = norm_ta * norm_B / norm_A
                # logger.info(f"Proportional scale factor for {key}: {scale_factor}")
            else:
                scale_factor = 1.0

            self.vector[key] = scale_factor * self.vector[key]

    def apply_to(self, pretrained_model, scaling_coef=1.0, layers_to_skip=None, bn_running_stats=True):
        """Apply a task vector to a pretrained model."""
        pretrained_state_dict = pretrained_model.state_dict()
        new_model = deepcopy(pretrained_model)
        with torch.no_grad():
            new_state_dict = {}

            for key in pretrained_state_dict:
                if 'num_batches_tracked' in key:
                    continue
                elif key not in self.vector:
                    logger.warning(f'Key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                # Check if the key corresponds to any of the layers we want to skip
                if layers_to_skip is not None:
                    if key in layers_to_skip:
                        new_state_dict[key] = pretrained_state_dict[key]
                    else:
                        new_state_dict[key] = pretrained_state_dict[key] + \
                            scaling_coef * self.vector[key]
                else:
                    new_state_dict[key] = pretrained_state_dict[key] + \
                        scaling_coef * self.vector[key]

                if not bn_running_stats:
                    if 'bn' and 'running' in key:  # only running stats
                        # if 'bn' in key: #all bns
                        new_state_dict[key] = pretrained_state_dict[key]

        new_model.load_state_dict(new_state_dict, strict=True)
        return new_model

    def apply_to_ptrim(self, model_new, model_old, scaling_coef=1.0, layers_to_skip=[], p=0.99, beta=0.0, method='linear'):
        """Apply a task vector to a pretrained model."""
        pretrained_state_dict = model_new.state_dict()
        new_model = deepcopy(model_new)
        with torch.no_grad():
            new_state_dict = {}

            for key in pretrained_state_dict:
                if 'identity' in key:
                    new_state_dict[key] = pretrained_state_dict[key]
                elif key not in self.vector:
                    logger.warning(f'Key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                # Check if the key corresponds to any of the layers we want to skip
                if len(layers_to_skip) > 0:
                    if any(k in key for k in layers_to_skip):
                        new_state_dict[key] = pretrained_state_dict[key]
                    else:
                        new_state_dict[key] = self.get_new_params(
                            pretrained_state_dict, key, model_new, model_old, scaling_coef, p, beta, method=method)
                else:
                    new_state_dict[key] = self.get_new_params(
                        pretrained_state_dict, key, model_new, model_old, scaling_coef, p, beta, method=method)

        new_model.load_state_dict(new_state_dict, strict=True)
        return new_model
