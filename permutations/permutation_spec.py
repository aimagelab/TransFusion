"""
Specification and construction of permutation mappings for model layers.
Defines how permutations are mapped to model layers and axes, and provides builders for different model architectures (ResNet, ViT, CLIP).

This module provides utility functions and builder classes to define how permutations (reorderings) of channels/neurons are mapped to the parameters of deep learning models. The permutation specifications are used for model merging, alignment, and research on permutation invariance.

All permutation axes are specified according to the storage order of weights in PyTorch (e.g., convolution weights are not transposed: (out_channels, in_channels, ...)).

All comments and docstrings are in English for clarity and research reproducibility.
"""
######################################################################
# Permutation Specification Classes
######################################################################
import copy
import logging
from collections import defaultdict
from functools import partial
from typing import NamedTuple

pylogger = logging.getLogger(__name__)


class PermutationSpec(NamedTuple):
    """
    Data structure specifying how permutations are mapped to model layers and axes.
    Attributes:
        perm_to_layers_and_axes (dict): Maps permutation names to the layers and axes they permute.
        layer_and_axes_to_perm (dict): Maps each layer and axis to the corresponding permutation (or None).
    """
    perm_to_layers_and_axes: dict
    layer_and_axes_to_perm: dict


class PermutationSpecBuilder:
    """
    Base class for building permutation specifications for different model architectures.
    """

    def __init__(self) -> None:
        pass

    def create_permutation_spec(self) -> list:
        """
        Should be implemented by subclasses to return a PermutationSpec for a specific architecture.
        """
        pass

    def permutation_spec_from_axes_to_perm(self, axes_to_perm: dict) -> PermutationSpec:
        """
        Utility to convert a mapping from axes to permutations into a PermutationSpec object.
        Args:
            axes_to_perm (dict): Mapping from layer names to tuples of permutation names (or None) per axis.
        Returns:
            PermutationSpec: The constructed permutation specification.
        """
        perm_to_axes = defaultdict(list)

        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))

        return PermutationSpec(perm_to_layers_and_axes=dict(perm_to_axes), layer_and_axes_to_perm=axes_to_perm)

######################################################################
# Utility functions for permutation specification
######################################################################


def conv_axes(name, in_perm, out_perm, bias=False):
    """
    Utility for specifying permutation axes for convolutional layers.
    Args:
        name (str): Layer name.
        in_perm (str or None): Permutation for input channels.
        out_perm (str or None): Permutation for output channels.
        bias (bool): Whether to include bias permutation.
    Returns:
        dict: Mapping from parameter names to permutation tuples.

    Note:
        The permutation spec is written this way because convolution weights are stored NOT transposed in PyTorch:
        the shape is (out_channels, in_channels, ...), so the first axis is output and the second is input.
    """
    axes = {f"{name}.weight": (out_perm, in_perm, None, None,)}

    if bias:
        axes[f"{name}.bias"] = (out_perm,)

    return axes

# Utility function for LayerNorm


def layernorm_axes(name, perm):
    """
    Utility for specifying permutation axes for LayerNorm layers.
    Args:
        name (str): Layer name.
        perm (str or None): Permutation for the normalized dimension.
    Returns:
        dict: Mapping from parameter names to permutation tuples.
    """
    return {f"{name}.weight": (perm,), f"{name}.bias": (perm,)}

# Utility function for BatchNorm


def batchnorm_axes(name, perm):
    """
    Utility for specifying permutation axes for BatchNorm layers.
    Args:
        name (str): Layer name.
        perm (str or None): Permutation for the normalized dimension.
    Returns:
        dict: Mapping from parameter names to permutation tuples.
    """
    return {
        f"{name}.weight": (perm,),
        f"{name}.bias": (perm,),
        f"{name}.running_mean": (perm,),
        f"{name}.running_var": (perm,),
        f"{name}.num_batches_tracked": (None,),
    }



def easyblock_axes(name, perm, norm_layer="ln"):
    """
    Utility for specifying permutation axes for a simple residual block (no channel change).
    Args:
        name (str): Block name.
        perm (str): Permutation for the block.
        norm_layer (str): Normalization layer type (default: "ln").
    Returns:
        dict: Mapping from parameter names to permutation tuples.
    """
    norm_axes = batchnorm_axes  # Always uses BatchNorm for normalization

    return {
        # First convolution and normalization
        **conv_axes(f"{name}.conv1", in_perm=perm, out_perm=f"P_{name}_inner1"),
        **norm_axes(f"{name}.bn1", f"P_{name}_inner1"),

        # Second convolution and normalization
        **conv_axes(f"{name}.conv2", in_perm=f"P_{name}_inner1", out_perm=f"P_{name}_inner2"),
        **norm_axes(f"{name}.bn2", f"P_{name}_inner2"),

        # Third convolution and normalization
        **conv_axes(f"{name}.conv3", in_perm=f"P_{name}_inner2", out_perm=perm),
        **norm_axes(f"{name}.bn3", perm),
    }



def shortcut_block_axes(name, in_perm, out_perm, norm_layer="ln"):
    """
    Utility for specifying permutation axes for a residual block with channel change (shortcut).
    Args:
        name (str): Block name.
        in_perm (str): Input permutation.
        out_perm (str): Output permutation.
        norm_layer (str): Normalization layer type (default: "ln").
    Returns:
        dict: Mapping from parameter names to permutation tuples.
    """
    norm_axes = batchnorm_axes  # Always uses BatchNorm for normalization

    return {
        # First convolution and normalization (input -> P_inner1)
        **conv_axes(f"{name}.conv1", in_perm=in_perm, out_perm=f"P_{name}_inner1"),
        **norm_axes(f"{name}.bn1", f"P_{name}_inner1"),

        # Second convolution and normalization (P_inner1 -> P_inner2)
        **conv_axes(f"{name}.conv2", in_perm=f"P_{name}_inner1", out_perm=f"P_{name}_inner2"),
        **norm_axes(f"{name}.bn2", f"P_{name}_inner2"),

        # Third convolution and normalization (P_inner2 -> output)
        **conv_axes(f"{name}.conv3", in_perm=f"P_{name}_inner2", out_perm=out_perm),
        **norm_axes(f"{name}.bn3", out_perm),

        # Shortcut: convolution to change channels from input to output
        **conv_axes(f"{name}.downsample.0", in_perm=in_perm, out_perm=out_perm),
        **norm_axes(f"{name}.downsample.1", out_perm),
    }

# Function to define dense layers


def dense_layer_axes(name, in_perm, out_perm, bias=True):
    """
    Utility for specifying permutation axes for dense (fully connected) layers.
    Args:
        name (str): Layer name.
        in_perm (str): Input permutation.
        out_perm (str): Output permutation.
        bias (bool): Whether to include bias permutation.
    Returns:
        dict: Mapping from parameter names to permutation tuples.
    """
    return {f"{name}.weight": (out_perm, in_perm), f"{name}.bias": (out_perm,)}

# Builder for the ResNet50 model, using 3 convolutions for each block


class ResNet50PermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for permutation specifications for ResNet50 architectures.
    """
    """
    Classe ResNet50 che definisce i permutazioni tra i layer, utilizzando 3 convoluzioni per blocco.
    """

    def __init__(self) -> None:
        pass

    def create_permutation_spec(self) -> PermutationSpec:
        return self.permutation_spec_from_axes_to_perm(
            {
                # Definition of permutations for logit scale (no permutation)
                "logit_scale": (None,),

                ########################
                ######## Visual ########
                ########################

                # Initial layers (pre-layer convolutions)
                **conv_axes("visual.conv1", in_perm=None, out_perm="P_conv1"),
                **batchnorm_axes("visual.bn1", "P_conv1"),

                **conv_axes("visual.conv2", in_perm="P_conv1", out_perm="P_conv2"),
                **batchnorm_axes("visual.bn2", "P_conv2"),

                **conv_axes("visual.conv3", in_perm="P_conv2", out_perm="P_conv3"),
                **batchnorm_axes("visual.bn3", "P_conv3"),

                # Layer 1 (easy block without channel change)
                **shortcut_block_axes("visual.layer1.0", in_perm="P_conv3", out_perm="P_conv4"),
                **easyblock_axes("visual.layer1.1", perm="P_conv4"),
                **easyblock_axes("visual.layer1.2", perm="P_conv4"),

                # Layer 2 (shortcut block with channel change)
                **shortcut_block_axes("visual.layer2.0", in_perm="P_conv4", out_perm="P_conv5"),
                **easyblock_axes("visual.layer2.1", perm="P_conv5"),
                **easyblock_axes("visual.layer2.2", perm="P_conv5"),
                **easyblock_axes("visual.layer2.3", perm="P_conv5"),

                # Layer 3
                **shortcut_block_axes("visual.layer3.0", in_perm="P_conv5", out_perm="P_conv6"),
                **easyblock_axes("visual.layer3.1", perm="P_conv6"),
                **easyblock_axes("visual.layer3.2", perm="P_conv6"),
                **easyblock_axes("visual.layer3.3", perm="P_conv6"),
                **easyblock_axes("visual.layer3.4", perm="P_conv6"),
                **easyblock_axes("visual.layer3.5", perm="P_conv6"),

                # Layer 4
                **shortcut_block_axes("visual.layer4.0", in_perm="P_conv6", out_perm="P_conv7"),
                **easyblock_axes("visual.layer4.1", perm="P_conv7"),
                **easyblock_axes("visual.layer4.2", perm="P_conv7"),

                # Attention Pool
                f'visual.attnpool.positional_embedding': (None, "P_conv7"),
                f'visual.attnpool.k_proj.weight': ('P_attn', "P_conv7"),
                f'visual.attnpool.k_proj.bias': ('P_attn',),
                f'visual.attnpool.q_proj.weight': ('P_attn', "P_conv7"),
                f'visual.attnpool.q_proj.bias': ('P_attn',),
                f'visual.attnpool.v_proj.weight': ('P_attn', "P_conv7"),
                f'visual.attnpool.v_proj.bias': ('P_attn',),
                f'visual.attnpool.c_proj.weight': ('P_out_proj', 'P_attn'),
                f'visual.attnpool.c_proj.bias': ('P_out_proj',),

                ########################
                ######## Text ##########
                ########################

                # Definizione dei permutazioni per il transformer
                **transformer_block_axes(12, p_in="P_in", p_out="P_last", tower="text"),

                # Token embedding
                "token_embedding.weight": (None, None),

                # Positional embedding
                "positional_embedding": (None, None),

                # Layer norm finale
                "ln_final.weight": (None,),
                "ln_final.bias": (None,),

                # Proiezione finale
                "text_projection": (None, None),
            }
        )


class CLIP_Visual_PermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for permutation specifications for CLIP visual transformer architectures.
    """

    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self) -> PermutationSpec:
        prefix = ''
        axes_to_perm = {
           
            **conv_axes(f"{prefix}conv1", in_perm=None, out_perm="P_conv"),
            f"{prefix}class_embedding": ("P_conv",),  # (dim)
            # (1, 1, dim) probably P_transf_in or its own P
            f"{prefix}positional_embedding": (None, "P_conv"),
            # (1, 1, dim) probably P_transf_in or its own P
            f"{prefix}ln_pre.weight": ("P_conv",),
            # (1, 1, dim) probably P_transf_in or its own P
            f"{prefix}ln_pre.bias": ("P_conv",),

            **transformer_block_axes_clip(self.depth, p_in="P_conv", p_out='P_last', prefix=prefix),
            f"{prefix}ln_post.weight": ('P_last',),
            f"{prefix}ln_post.bias": ('P_last',),
            # (1, 1, dim) probably P_transf_in or its own P
            f"{prefix}proj": ('P_last', None),  # Note: this weights are not transposed, it's also possible to permute the output of the visual encoder


        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class CLIP_Text_PermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for permutation specifications for CLIP text transformer architectures.
    """

    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self) -> PermutationSpec:
        prefix = ''
        axes_to_perm = {
            # token embedding
            "token_embedding.weight": (None, "P_T_in"),  # ()?

            # positional_embedding
            "positional_embedding": (None, "P_T_in"),  # ()?

            # transformer
            # **transformer_block_axes(self.depth, p_in="P_in", p_out="P_last", tower="text"),
            **transformer_block_axes_clip(self.depth, p_in="P_T_in", p_out='P_T_last'),

            # layer norm
            "ln_final.weight": ('P_T_last',),  # ()?
            "ln_final.bias": ('P_T_last',),  # ()?

            # linear proj
            "text_projection": ('P_T_last', None),  # Note: this weights are not transposed,  it's also possible to permute the output of the text encoder

        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


def transformer_block_axes_clip(depth, p_in, p_out, prefix=''):
    """
    Utility for specifying permutation axes for CLIP transformer blocks.
    Args:
        depth (int): Number of transformer blocks.
        p_in (str): Input permutation.
        p_out (str): Output permutation.
        prefix (str): Optional prefix for parameter names.
    Returns:
        dict: Mapping from parameter names to permutation tuples for all blocks.
    """

    all_axes = {}

    for block_ind in range(depth):

        block_out = p_out if block_ind == depth - 1 else f"P{block_ind}_out"
        block_in = p_in if block_ind == 0 else f"P{block_ind-1}_out"

        block_axes = {
            # --- Attention Block ---
            # LayerNorm before attention (shape: [dim])
            f"{prefix}transformer.resblocks.{block_ind}.ln_1.weight": (block_in,),
            f"{prefix}transformer.resblocks.{block_ind}.ln_1.bias": (block_in,),

            # Attention heads: Q, K, V projections (shape: [head_dim, dim])
            # Note: weights are not transposed in PyTorch, so output comes first
            f"{prefix}transformer.resblocks.{block_ind}.attn.q.weight": (f"P{block_ind}_attn_QK", block_in),
            f"{prefix}transformer.resblocks.{block_ind}.attn.k.weight": (f"P{block_ind}_attn_QK", block_in),
            f"{prefix}transformer.resblocks.{block_ind}.attn.v.weight": (f"P{block_ind}_attn_QK", block_in),

            # Biases for Q, K, V projections (shape: [head_dim])
            f"{prefix}transformer.resblocks.{block_ind}.attn.q.bias": (f"P{block_ind}_attn_QK",),
            f"{prefix}transformer.resblocks.{block_ind}.attn.k.bias": (f"P{block_ind}_attn_QK",),
            f"{prefix}transformer.resblocks.{block_ind}.attn.v.bias": (f"P{block_ind}_attn_QK",),

            # Output projection after attention (shape: [dim, head_dim])
            f"{prefix}transformer.resblocks.{block_ind}.attn.proj.weight": (f"P{block_ind}_out_proj", f"P{block_ind}_attn_QK"),
            # Bias for output projection (shape: [dim])
            f"transformer.resblocks.{block_ind}.attn.proj.bias": (f"P{block_ind}_out_proj",),

            # Residual shortcut for attention (shape: [dim, dim])
            f"transformer.resblocks.{block_ind}.attn.shortcut_1.identity": (f"P{block_ind}_out_proj", block_in),

            # --- MLP Block ---
            # LayerNorm before MLP (shape: [dim])
            f"{prefix}transformer.resblocks.{block_ind}.ln_2.weight": (f"P{block_ind}_out_proj",),
            f"{prefix}transformer.resblocks.{block_ind}.ln_2.bias": (f"P{block_ind}_out_proj",),

            # First linear layer in MLP (shape: [mlp_dim, dim])
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc1.weight": (
                f"P{block_ind}_mlp_out",
                f"P{block_ind}_out_proj",
            ),
            # Bias for first MLP linear (shape: [mlp_dim])
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc1.bias": (f"P{block_ind}_mlp_out",),

            # Second linear layer in MLP (shape: [dim, mlp_dim])
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc2.weight": (block_out, f"P{block_ind}_mlp_out",),
            # Bias for second MLP linear (shape: [dim])
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc2.bias": (block_out,),

            # Residual shortcut for MLP (shape: [dim, dim])
            f"transformer.resblocks.{block_ind}.mlp.shortcut_2.identity": (block_out, f"P{block_ind}_out_proj"),
        }
        all_axes.update(block_axes)

    return all_axes


class ViTPermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for permutation specifications for generic Vision Transformer (ViT) architectures.
    """

    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:

        axes_to_perm = {
            # layer norm
            "to_patch_embedding.to_patch_tokens.1.weight": (None,),  # (3*c*16)
            "to_patch_embedding.to_patch_tokens.1.bias": (None,),  # (3*c*16)
            # linear
            # (dim, 3*c*16)
            "to_patch_embedding.to_patch_tokens.2.weight": ("P_in", None),
            "to_patch_embedding.to_patch_tokens.2.bias": ("P_in",),  # dim
            # (1, p+1, dim) probably P_transf_in or its own P
            "pos_embedding": (None, None, "P_in"),
            # (1, 1, dim) probably P_transf_in or its own P
            "cls_token": (None, None, "P_in"),
            **transformer_block_axes(self.depth, p_in="P_in", p_out="P_last"),
            # layer norm
            "mlp_head.0.weight": ("P_last",),  # (dim, )
            "mlp_head.0.bias": ("P_last",),  # (dim,)
            # linear
            "mlp_head.1.bias": (None,),  # (num_classes)
            "mlp_head.1.weight": (None, "P_last"),  # (num_classes, dim)
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


def transformer_block_axes(depth, p_in, p_out):
    """
    Utility for specifying permutation axes for generic transformer blocks.
    Args:
        depth (int): Number of transformer blocks.
        p_in (str): Input permutation.
        p_out (str): Output permutation.
    Returns:
        dict: Mapping from parameter names to permutation tuples for all blocks.
    """

    all_axes = {}

    for block_ind in range(depth):

        block_out = p_out if block_ind == depth - 1 else f"P{block_ind}_out"
        block_in = p_in if block_ind == 0 else f"P{block_ind-1}_out"
        # block_out = None
        # block_in = None

        block_axes = {
            # Attention
            # layer norm
            # (dim,)
            f"transformer.layers.{block_ind}.0.norm.weight": (block_in,),
            # (dim,)
            f"transformer.layers.{block_ind}.0.norm.bias": (block_in,),
            f"transformer.layers.{block_ind}.0.temperature": (None,),  # (,)
            # HEADS
            # (head_dim, dim)
            f"transformer.layers.{block_ind}.0.to_q.weight": (f"P{block_ind}_attn_QK", block_in),
            # (head_dim, dim)
            f"transformer.layers.{block_ind}.0.to_k.weight": (f"P{block_ind}_attn_QK", block_in),
            # (head_dim, dim)
            f"transformer.layers.{block_ind}.0.to_v.weight": (f"P{block_ind}_attn_QK", block_in),
            # Attention out
            # (dim, dim)
            f"transformer.layers.{block_ind}.0.to_out.0.weight": (f"P{block_ind}_out_proj", f"P{block_ind}_attn_QK"),
            # (dim,)
            f"transformer.layers.{block_ind}.0.to_out.0.bias": (f"P{block_ind}_out_proj",),
            # shortcut
            # (dim, dim) # WORKS
            f"transformer.layers.{block_ind}.1.identity": (f"P{block_ind}_out_proj", block_in),
            # MLP
            # layer norm
            # (dim, )
            f"transformer.layers.{block_ind}.2.net.0.weight": (f"P{block_ind}_out_proj",),
            # (dim,)
            f"transformer.layers.{block_ind}.2.net.0.bias": (f"P{block_ind}_out_proj",),
            # linear 1
            # (mlp_dim, dim)
            f"transformer.layers.{block_ind}.2.net.1.weight": (f"P{block_ind}_mlp_out", f"P{block_ind}_out_proj",),
            # (mlp_dim,)
            f"transformer.layers.{block_ind}.2.net.1.bias": (f"P{block_ind}_mlp_out",),
            # linear 2
            # (dim, mlp_dim) # WORKS
            f"transformer.layers.{block_ind}.2.net.4.weight": (block_out, f"P{block_ind}_mlp_out",),
            # (dim,) # WORKS
            f"transformer.layers.{block_ind}.2.net.4.bias": (block_out,),
            # shortcut 2
            # (dim, dim) # WORKS
            f"transformer.layers.{block_ind}.3.identity": (block_out, f"P{block_ind}_out_proj"),
        }

        all_axes.update(block_axes)

    return all_axes


class ViT_B_PermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for permutation specifications for ViT-B architectures.
    """

    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:
        axes_to_perm = {
            # Embedding layers
            "vit.embeddings.cls_token": (None, None, "P_in"),  # (1, 1, dim)
            # (1, num_patches+1, dim)
            "vit.embeddings.position_embeddings": (None, None, "P_in"),
            # (dim, channels, kernel_h, kernel_w)
            "vit.embeddings.patch_embeddings.projection.weight": ("P_in",),
            # (dim)
            "vit.embeddings.patch_embeddings.projection.bias": ("P_in",),

            # Encoder Layers
            **self.vit_b_transformer_block_axes(self.depth, p_in="P_in", p_out="P_out"),

            # Final Layers
            "vit.layernorm.weight": ("P_out",),  # (dim)
            "vit.layernorm.bias": ("P_out",),  # (dim)
            "classifier.weight": (None, "P_out"),  # (num_classes, dim)
            "classifier.bias": (None,),  # (num_classes)
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)

    def vit_b_transformer_block_axes(self, depth, p_in, p_out):
        all_axes = {}

        for layer_idx in range(depth):
            block_in = p_in if layer_idx == 0 else f"P{layer_idx-1}_out"
            block_out = p_out if layer_idx == depth - \
                1 else f"P{layer_idx}_out"

            block_axes = {

                # Layer Norm before attention
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_before.weight": (block_in,),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_before.bias": (block_in,),

                # Attention: Query, Key, Value
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.query.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.query.bias": (f"P{layer_idx}_attn",),
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.key.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.key.bias": (f"P{layer_idx}_attn",),
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.value.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.value.bias": (f"P{layer_idx}_attn",),

                # Attention output
                # (dim, head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.output.dense.weight": (f"P{layer_idx}_dense_out", f"P{layer_idx}_attn"),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.attention.output.dense.bias": (f"P{layer_idx}_dense_out",),

                # Attention output shortcut
                # (dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.shortcut_1.identity": (f"P{layer_idx}_dense_out", block_in),

                # Layer Norm after attention and residual
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_after.weight": (f"P{layer_idx}_dense_out",),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_after.bias": (f"P{layer_idx}_dense_out",),

                # MLP Layers
                # (mlp_dim, dim)
                f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight": (f"P{layer_idx}_mlp", f"P{layer_idx}_dense_out"),
                # (mlp_dim)
                f"vit.encoder.layer.{layer_idx}.intermediate.dense.bias": (f"P{layer_idx}_mlp",),
                # (dim, mlp_dim)
                f"vit.encoder.layer.{layer_idx}.output.dense.weight": (block_out, f"P{layer_idx}_mlp"),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.output.dense.bias": (block_out,),

                # MLP output shortcut
                # (dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.shortcut_2.identity": (block_out, f"P{layer_idx}_dense_out"),

            }

            all_axes.update(block_axes)

        return all_axes


class Naive_ViT_B_PermutationSpecBuilder(PermutationSpecBuilder):
    """
    Builder for naive permutation specifications for ViT-B architectures (for ablation or baseline).
    """

    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:
        axes_to_perm = {
            # Embedding layers
            "vit.embeddings.cls_token": (None, None, "P_in"),  # (1, 1, dim)
            # (1, num_patches+1, dim)
            "vit.embeddings.position_embeddings": (None, None, "P_in"),
            # (dim, channels, kernel_h, kernel_w)
            "vit.embeddings.patch_embeddings.projection.weight": ("P_in",),
            # (dim)
            "vit.embeddings.patch_embeddings.projection.bias": ("P_in",),

            # Encoder Layers
            **self.naive_vit_b_transformer_block_axes(self.depth, p_in="P_in", p_out="P_out"),

            # Final Layers
            "vit.layernorm.weight": ("P_out",),  # (dim)
            "vit.layernorm.bias": ("P_out",),  # (dim)
            "classifier.weight": (None, "P_out"),  # (num_classes, dim)
            "classifier.bias": (None,),  # (num_classes)
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)

    def naive_vit_b_transformer_block_axes(self, depth, p_in, p_out):
        all_axes = {}

        for layer_idx in range(depth):
            block_in = p_in if layer_idx == 0 else f"P{layer_idx-1}_out"
            block_out = p_out if layer_idx == depth - \
                1 else f"P{layer_idx}_out"

            block_axes = {

                # Layer Norm before attention
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_before.weight": (block_in,),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_before.bias": (block_in,),

                # Attention: Query, Key, Value
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.query.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.query.bias": (f"P{layer_idx}_attn",),
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.key.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.key.bias": (f"P{layer_idx}_attn",),
                # (head_dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.value.weight": (f"P{layer_idx}_attn", block_in),
                # (head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.attention.value.bias": (f"P{layer_idx}_attn",),

                # Attention output
                # (dim, head_dim)
                f"vit.encoder.layer.{layer_idx}.attention.output.dense.weight": (f"P{layer_idx}_dense_out", f"P{layer_idx}_attn"),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.attention.output.dense.bias": (f"P{layer_idx}_dense_out",),

                # Attention output shortcut
                # (dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.shortcut_1.identity": (None, None),

                # Layer Norm after attention and residual
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_after.weight": (f"P{layer_idx}_dense_out",),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.layernorm_after.bias": (f"P{layer_idx}_dense_out",),

                # MLP Layers
                # (mlp_dim, dim)
                f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight": (f"P{layer_idx}_mlp", f"P{layer_idx}_dense_out"),
                # (mlp_dim)
                f"vit.encoder.layer.{layer_idx}.intermediate.dense.bias": (f"P{layer_idx}_mlp",),
                # (dim, mlp_dim)
                f"vit.encoder.layer.{layer_idx}.output.dense.weight": (block_out, f"P{layer_idx}_mlp"),
                # (dim)
                f"vit.encoder.layer.{layer_idx}.output.dense.bias": (block_out,),

                # MLP output shortcut
                # (dim, dim)
                f"vit.encoder.layer.{layer_idx}.attention.shortcut_2.identity": (None, None),

            }

            all_axes.update(block_axes)

        return all_axes
