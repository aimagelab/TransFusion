"""
Model definitions and wrappers for OpenCLIP and LoRA.


This module extends the standard OpenCLIP vision encoder with custom layers and logic for advanced research purposes.

Key modifications:
- Adds a custom Shortcut layer (parameterized identity matrix) to enable feature space permutations. This allows for channel swapping, model merging, permutation alignment, and other advanced manipulations. The shortcut can be replaced with a permutation matrix for these purposes.
- Includes a DiagLayerNorm class, a diagonal-matrix version of LayerNorm that allows the scale parameter (gamma) to be a matrix. While not currently used in the default model, it is provided for potential research on permutation-invariant normalization and advanced model merging strategies.

These changes make the model suitable for research on model merging, permutation invariance, and advanced fine-tuning strategies, including both LoRA-style and full-rank adaptation (see lora_utils.py for details on AB argument usage).

Adapted from: https://github.com/mlfoundations/open_clip
Requires: pip install open_clip_torch

Note:
    A list of available models and results for various datasets can be found at:
    https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
"""
######################################################################
# Utility Classes and Functions
######################################################################

import logging
import math
import types
from typing import Dict, List, Tuple, Union
import open_clip.transformer
import torch
from torch import Size
import torch.nn as nn

# from utils import binary_to_boolean_type
try:
    import open_clip
except ImportError:
    raise ImportError(
        "Please install the OpenCLIP package by running: `pip install open_clip_torch`")

from src.lora_utils import LoRAAttention, LoRAMlp
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Shortcut(nn.Module):
    """
    Identity shortcut layer for residual connections.
    This layer can be replaced with a permutation matrix for feature space manipulation.
    """

    def __init__(self, dim):
        super().__init__()
        # TODO: should have requires_grad=False
        self.identity = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, x):
        """
        Applies the (optionally permuted) identity matrix to the input.
        Args:
            x (Tensor): Input tensor of shape [..., dim].
        Returns:
            Tensor: Output tensor after applying the shortcut.
        """
        return x @ self.identity.T



class DiagLayerNorm(nn.LayerNorm):
    """
    LayerNorm variant that applies a diagonal weight matrix instead of elementwise scaling.
    This allows permutations to be applied to both activations and normalization parameters.

    Example usage for a single transformer block:
        block.ln_1 = DiagLayerNorm(
            normalized_shape=model.visual.transformer.width,
            eps=block.ln_1.eps,
            elementwise_affine=True,
            weight=block.ln_1.weight,
            bias=block.ln_1.bias
        )
        block.ln_2 = DiagLayerNorm(
            normalized_shape=model.visual.transformer.width,
            eps=block.ln_2.eps,
            elementwise_affine=True,
            weight=block.ln_2.weight,
            bias=block.ln_2.bias
        )
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, weight: torch.Tensor = None, bias: torch.Tensor = None, **kwargs) -> None:
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            **kwargs
        )
        if weight is not None:
            self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)

        if self.elementwise_affine:
            # Convert the original weight (vector) to a diagonal matrix for matrix multiplication
            self.weight = nn.Parameter(torch.diag(self.weight))  # Shape [C, C]

    def forward(self, x: torch.Tensor):
        """
        Applies diagonal LayerNorm to the input.
        Args:
            x (Tensor): Input tensor of shape [..., C].
        Returns:
            Tensor: Normalized and scaled tensor.
        """
        orig_type = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)

        # Use matrix multiplication with diagonal weight instead of elementwise scaling
        if self.elementwise_affine:
            x = x @ self.weight.T  # Shape [..., C]
            x = x + self.bias  # Add bias
        return x.to(orig_type)


def _expand_token(token, batch_size: int):
    """
    Expands a token embedding to match the batch size.
    Args:
        token (Tensor): Token embedding of shape [1, 1, dim].
        batch_size (int): Desired batch size.
    Returns:
        Tensor: Expanded token embedding of shape [batch_size, 1, dim].
    """
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def forward_visual(ext, x: torch.Tensor, AB: dict = None):
    """
    Forward pass for the visual encoder, including patch embedding, positional encoding,
    transformer, and pooling. Handles both standard and contrastive pooling.
    Args:
        ext (nn.Module): Visual encoder module.
        x (Tensor): Input image tensor.
        AB (dict, optional): Additional bias for advanced adaptation (LoRA, etc.).
    Returns:
        Tensor or Tuple: Pooled features or (pooled, tokens) if output_tokens is True.
    """
    x = ext.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    x = torch.cat([_expand_token(ext.class_embedding,
                  x.shape[0]).to(x.dtype), x], dim=1)
    # shape = [*, grid ** 2 + 1, width]
    x = x + ext.positional_embedding.to(x.dtype)

    x = ext.patch_dropout(x)
    x = ext.ln_pre(x)
    x = ext.transformer(x, AB=AB)

    if ext.attn_pool is not None:
        if ext.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = ext.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = ext.attn_pool(x)
            if ext.attn_pool_type == 'parallel':
                pooled = ext.attn_pool_contrastive(x)
            else:
                assert ext.attn_pool_type == 'cascade'
                pooled = ext.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = ext.attn_pool(x)
            x = ext.ln_post(x)
            pooled, tokens = ext._global_pool(x)
    elif ext.final_ln_after_pool:
        pooled, tokens = ext._global_pool(x)
        pooled = ext.ln_post(pooled)
    else:
        x = ext.ln_post(x)
        pooled, tokens = ext._global_pool(x)

    if ext.proj is not None:
        pooled = pooled @ ext.proj

    if ext.output_tokens:
        return pooled, tokens

    return pooled


def transformer_forward(ext, x: torch.Tensor, attn_mask=None, AB=None):
    """
    Forward pass for the transformer encoder, iterating over residual blocks.
    Args:
        ext (nn.Module): Transformer module.
        x (Tensor): Input tensor.
        attn_mask (Tensor, optional): Attention mask.
        AB (dict, optional): Additional bias for advanced adaptation.
    Returns:
        Tensor: Output of the transformer encoder.
    """
    if not ext.batch_first:
        x = x.transpose(0, 1).contiguous()    # NLD -> LND
    for i, r in enumerate(ext.resblocks):
        ab_l = AB.get(i) if AB is not None else None
        x = r(x, attn_mask=attn_mask, AB=ab_l)
    if not ext.batch_first:
        x = x.transpose(0, 1)    # LND -> NLD
    return x


def attention(ext, q_x: torch.Tensor, k_x=None, v_x=None, attn_mask=None, AB=None):
    """
    Compute attention using the provided query, key, and value tensors.
    If k_x or v_x are not provided, use q_x for self-attention.
    Optionally supports additional bias (AB) for advanced attention mechanisms.
    Args:
        ext (nn.Module): Attention module.
        q_x (Tensor): Query tensor.
        k_x (Tensor, optional): Key tensor.
        v_x (Tensor, optional): Value tensor.
        attn_mask (Tensor, optional): Attention mask.
        AB (Tensor, optional): Additional bias.
    Returns:
        Tensor: Output of the attention mechanism.
    """
    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x

    attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
    if AB:
        return ext.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask, AB=AB
        )[0]
    else:
        return ext.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]


def block_forward(ext, q_x: torch.Tensor, k_x=None, v_x=None, attn_mask=None, AB=None):
    """
    Forward pass for a transformer block, including attention and MLP with shortcuts.
    Args:
        ext (nn.Module): Transformer block module.
        q_x (Tensor): Query tensor.
        k_x (Tensor, optional): Key tensor.
        v_x (Tensor, optional): Value tensor.
        attn_mask (Tensor, optional): Attention mask.
        AB (Tensor, optional): Additional bias.
    Returns:
        Tensor: Output of the transformer block.
    """
    k_x = ext.ln_1_kv(k_x) if hasattr(
        ext, "ln_1_kv") and k_x is not None else None
    v_x = ext.ln_1_kv(v_x) if hasattr(
        ext, "ln_1_kv") and v_x is not None else None

    x = ext.attn.shortcut_1(q_x) + ext.ls_1(ext.attention(q_x=ext.ln_1(q_x),
                                                          k_x=k_x, v_x=v_x, attn_mask=attn_mask, AB=AB))
    x = ext.mlp.shortcut_2(x) + ext.ls_2(ext.mlp(ext.ln_2(x)))
    return x


@torch.no_grad()
def setup_visual(model: nn.Module):
    """
    Modifies the visual transformer of the model to use custom forward methods, LoRA layers,
    and diagonal LayerNorm. This enables advanced fine-tuning and adaptation.
    Args:
        model (nn.Module): The model whose visual transformer will be patched.
    """
    device = next(iter(model.parameters())).device
    # for param in model.parameters():
    #     param.requires_grad_(False)

    model.visual.forward = types.MethodType(forward_visual, model.visual)
    model.visual.transformer.forward = types.MethodType(
        transformer_forward, model.visual.transformer)

    # Visual and text
    # for model in [model.visual.transformer.resblocks, model.transformer.resblocks]:
    # visual
    for block in model.visual.transformer.resblocks:
        block.forward = types.MethodType(block_forward, block)
        block.attention = types.MethodType(attention, block)
        # customize layernorm


        # replace attention
        dim = block.attn.embed_dim
        n_heads = block.attn.num_heads
        qkv_bias = block.attn.in_proj_bias is not None
        proj_bias = block.attn.out_proj.bias is not None
        attn_drop = block.attn.dropout
        new_attn = LoRAAttention(
            dim, n_heads, attn_drop=attn_drop, qkv_bias=qkv_bias, proj_bias=proj_bias).to(device)
        new_attn.q.weight.data = block.attn.in_proj_weight[:dim]
        new_attn.k.weight.data = block.attn.in_proj_weight[dim:2 * dim]
        new_attn.v.weight.data = block.attn.in_proj_weight[2 * dim:3 * dim]
        if qkv_bias:
            new_attn.q.bias.data = block.attn.in_proj_bias[:dim]
            new_attn.k.bias.data = block.attn.in_proj_bias[dim:2 * dim]
            new_attn.v.bias.data = block.attn.in_proj_bias[2 * dim:3 * dim]
        new_attn.proj.weight.data = block.attn.out_proj.weight.data
        if proj_bias:
            new_attn.proj.bias.data = block.attn.out_proj.bias.data
        new_attn.shortcut_1 = Shortcut(dim).to(device)
        block.attn = new_attn

        # replace mlp
        in_features = block.mlp.c_fc.in_features
        out_features = block.mlp.c_proj.out_features
        hidden_features = block.mlp.c_fc.out_features
        new_mlp = LoRAMlp(in_features, hidden_features,
                          out_features, bias=block.mlp.c_fc.bias is not None).to(device)
        new_mlp.fc1.weight.data.zero_()
        new_mlp.fc1.weight.data.add_(block.mlp.c_fc.weight)
        if block.mlp.c_fc.bias is not None:
            new_mlp.fc1.bias.data.zero_()
            new_mlp.fc1.bias.data.add_(block.mlp.c_fc.bias)

        new_mlp.fc2.weight.data.zero_()
        new_mlp.fc2.weight.data.add_(block.mlp.c_proj.weight)
        if block.mlp.c_proj.bias is not None:
            new_mlp.fc2.bias.data.zero_()
            new_mlp.fc2.bias.data.add_(block.mlp.c_proj.bias)
        new_mlp.shortcut_2 = Shortcut(dim).to(device)
        block.mlp = new_mlp


######################################################################
# Main Model Wrapper
######################################################################

class OpenCLIPModel(nn.Module):
    """
    Wrapper for OpenCLIP models, providing unified image/text encoding and custom forward logic.
    Automatically patches the visual transformer for advanced fine-tuning and adaptation.
    """
    @torch.no_grad()
    def __init__(self, clip_model: open_clip.CLIP, args=None) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.args = args
        setup_visual(self.clip_model)

    def encode_image(self, image):
        """
        Encode an image using the visual encoder.
        Args:
            image (Tensor): Input image tensor.
        Returns:
            Tensor: Encoded image features.
        """
        return self.clip_model.visual(image)

    def encode_text(self, text):
        """
        Encode text using the transformer encoder.
        Args:
            text (Tensor): Input text tensor.
        Returns:
            Tensor: Encoded text features.
        """
        return self.clip_model.transformer(text)

    def forward(
            self,
            image: torch.Tensor = None,
            text: torch.Tensor = None,
    ):
        """
        Forward pass for the OpenCLIP model.
        Returns encoded image and text features, and the logit scale (and optionally logit bias).

        Args:
            image (torch.Tensor, optional): Input image tensor.
            text (torch.Tensor, optional): Input text tensor.

        Returns:
            Tuple or dict: Encoded features and scaling parameters, depending on output_dict and logit_bias.
        """
        image_features = self.clip_model.encode_image(
            image, normalize=True) if image is not None else None
        text_features = self.clip_model.encode_text(
            text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.clip_model.logit_scale.exp()
            }
            if self.clip_model.logit_bias is not None:
                out_dict['logit_bias'] = self.clip_model.logit_bias
            return out_dict

        if self.clip_model.logit_bias is not None:
            return image_features, text_features, self.clip_model.logit_scale.exp(), self.clip_model.logit_bias
        return image_features, text_features, self.clip_model.logit_scale.exp()
