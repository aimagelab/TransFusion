"""
Parameter utilities for both LoRA (Low-Rank Adaptation) and full-rank (standard) fine-tuning.
Despite the name, this module supports both low-rank (LoRA-style) and full-rank (classic) parameter adaptation.

The AB argument in the custom layers can be used to inject either a low-rank delta (LoRA) or a full-rank delta (classic fine-tuning), depending on its structure:
- If AB is a dict with both 'A' and 'B', a low-rank (LoRA) update is applied: output += B @ (A @ x^T)
- If AB is a tensor or a dict with only 'B', a full-rank delta is applied: output += B @ x^T

This allows the same code to be used for both LoRA and full fine-tuning scenarios.
"""
######################################################################
# LoRA Parameter Utilities
######################################################################
import collections.abc
from itertools import repeat
import math
from torch import nn
import torch
import torch.nn.functional as F


def get_parameter(shape, device='cpu', type_init: str = 'kaiming', transpose: bool = False) -> torch.nn.Parameter:
    """
    Get a parameter with the specified shape and initialization

    Args:
        shape: tuple with the shape of the parameter
        device: device where the parameter will be allocated
        type_init: initialization type
        transpose: if True, the parameter will be transposed

    Returns:
        torch.nn.Parameter: parameter with the specified shape and initialization
    """

    param = torch.zeros(*shape, dtype=torch.float32, device=device)
    if type_init == 'orto':
        torch.nn.init.orthogonal_(param)
    if type_init == 'gaussian':
        torch.nn.init.normal_(param, mean=0.0, std=0.1)
    if type_init == 'kernel':
        torch.nn.init.normal_(param, mean=0.0, std=0.036)
    if type_init == 'attn':
        torch.nn.init.normal_(param, mean=1.0, std=0.03)
    if type_init == 'kaiming':
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    if type_init == 'ones':
        torch.nn.init.ones_(param)
    if transpose:
        param = torch.transpose(param, 1, 2)
    return torch.nn.Parameter(param)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class LoRALayer():
    def __init__(
            self,
            lora_dropout: float,
    ):
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoRALinear(nn.Linear, LoRALayer):
    """
    Linear layer supporting both LoRA-style (low-rank) and full-rank adaptation.
    If AB is provided, applies an additional delta to the output:
    - If AB contains both 'A' and 'B', applies a low-rank (LoRA) update: output += B @ (A @ x^T)
    - If AB contains only 'B' or is a tensor, applies a full-rank delta: output += B @ x^T
    If AB is None, behaves as a standard nn.Linear.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = True
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor, AB: dict = None):

        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            A = None
            if isinstance(AB, dict):
                B = AB['B']
                A = AB.get('A')
            else:
                B = AB
            if A is not None:
                return result + (B @ (A @ x.transpose(1, 2).unsqueeze(1))).sum(1).transpose(1, 2)
            return result + (B @ x.transpose(1, 2).unsqueeze(1)).sum(1).transpose(1, 2)

        return result


class LoRAAttention(nn.Module):
    """
    Attention layer as used in Vision Transformer.
    Supports both LoRA-style (low-rank) and full-rank adaptation via the AB argument.
    See LoRALinear for details on AB usage.
    """
    """
    Attention layer as used in Vision Transformer.
    Adapted to support LoRA-style parameters.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to q, k, v
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate after the final projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.embed_dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = LoRALinear(dim, dim, 0., bias=qkv_bias)
        self.k = LoRALinear(dim, dim, 0., bias=qkv_bias)
        self.v = LoRALinear(dim, dim, 0., bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LoRALinear(dim, dim, 0., bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv, AB: dict = None, **kwargs):
        """
        Forward pass of the attention layer.
        Supports `AB` for LoRA-style parameters (checkout docs for `VisionTransformer.forward`).

        Args:
            x: Input tensor
            AB: Dictionary containing LoRA-style parameters for the layer
        """

        B, N, C = xq.shape

        AB_q, AB_k, AB_v = None, None, None
        if AB is not None:
            AB_q = AB.get("q")
            AB_k = AB.get("k")
            AB_v = AB.get("v")

        q = self.q(xq, AB_q)
        k = self.k(xk, AB_k)
        v = self.v(xv, AB_v)
        q = q.reshape(B, N, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3)

        # NOTE: flash attention is less debuggable than the original. Use the commented code below if in trouble.
        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(
                q, k, v, scale=1 / math.sqrt(q.shape[-1]), dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)

        AB_proj = None

        if AB is not None:
            AB_proj = AB.get("proj")

        x = self.proj(x, AB_proj)
        x = self.proj_drop(x)

        return x, None  # None is for the attention weights


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LoRAMlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    Supports both LoRA-style (low-rank) and full-rank adaptation via the AB argument.
    See LoRALinear for details on AB usage.
    """
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    Adapted to support LoRA-style parameters.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        assert use_conv is False

        self.fc1 = LoRALinear(in_features, hidden_features,
                              bias=bias[0], lora_dropout=0.)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = LoRALinear(hidden_features, out_features,
                              bias=bias[1], lora_dropout=0.)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor, AB: dict = None, **kwargs):
        """
        Forward pass of the MLP layer.
        Supports `AB` for LoRA-style parameters (checkout docs for `VisionTransformer.forward`).

        Args:
            x: Input tensor
            AB: Dictionary containing LoRA-style parameters for the layer
        """
        AB_fc1 = None
        AB_fc2 = None

        if AB is not None:
            AB_fc1 = AB.get("fc1")
            AB_fc2 = AB.get("fc2")

        x = self.fc1(x, AB_fc1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, AB_fc2)
        x = self.drop2(x)

        return x


class LoRAParamModel(torch.nn.Module):

    def __init__(self, args, embed_dim: int, mlp_ratio: int, n_layers=12, rank=16):

        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.rank = rank

        self.n_layers = n_layers

        self.lora_config = self.build_lora_config()

        for l in range(self.n_layers):

            AB_q = self.create_AB(
                self.embed_dim, self.embed_dim, r1=rank, r2=rank)
            AB_k = self.create_AB(
                self.embed_dim, self.embed_dim, r1=rank, r2=rank)
            AB_v = self.create_AB(
                self.embed_dim, self.embed_dim, r1=rank, r2=rank)
            setattr(self, f'A_q_{l}', AB_q[0])
            setattr(self, f'B_q_{l}', AB_q[1])
            setattr(self, f'A_k_{l}', AB_k[0])
            setattr(self, f'B_k_{l}', AB_k[1])
            setattr(self, f'A_v_{l}', AB_v[0])
            setattr(self, f'B_v_{l}', AB_v[1])

            AB_proj = self.create_AB(self.embed_dim, self.embed_dim)

            setattr(self, f'A_proj_{l}', AB_proj[0])
            setattr(self, f'B_proj_{l}', AB_proj[1])

            AB_fc1 = self.create_AB(
                self.embed_dim, int(mlp_ratio * self.embed_dim))

            setattr(self, f'A_fc1_{l}', AB_fc1[0])
            setattr(self, f'B_fc1_{l}', AB_fc1[1])

            AB_fc2 = self.create_AB(
                int(mlp_ratio * self.embed_dim), self.embed_dim)

            setattr(self, f'A_fc2_{l}', AB_fc2[0])
            setattr(self, f'B_fc2_{l}', AB_fc2[1])

    def create_AB(self, fin, fout, r1=None, r2=None, transpose=False):
        r1 = self.rank if r1 is None else r1
        r2 = self.rank if r2 is None else r2

        config = ('kaiming', 'zeros')

        return get_parameter((1, r1, fin), type_init=config[0], transpose=transpose), get_parameter((1, fout, r2), type_init=config[1], transpose=transpose)

    def _get_matrix(self, namevar, layer_idx):
        return getattr(self, f'{namevar}_{layer_idx}')

    def get_lora_matrices(self, task_weights=None):
        return {
            layer_idx: self.get_lora_matrices_by_layer(
                layer_idx, task_weights=task_weights[layer_idx] if task_weights is not None else None)
            for layer_idx in range(self.n_layers)
        }

    def get_lora_matrices_by_layer(self, layer_idx, task_weights=None):

        params_dict = {
            loravar: self._gather_matrices(layer_idx, loravar) for loravar in self.lora_config
        }

        m = {}

        for op in ['q', 'k', 'v', 'proj', 'fc1', 'fc2']:

            A_op, B_op = f'A_{op}', f'B_{op}'

            A_m = params_dict[A_op][0]
            B_m = params_dict[B_op][0]

            w = 1
            if task_weights is not None:
                w = task_weights[:, None, None]
            m[op] = {
                "B": B_m,
                "A": A_m * w
            }

        return m

    def _gather_matrices(self, layer_idx: int, namevar: str):
        return self._get_matrix(namevar, layer_idx)

    def build_lora_config(self):
        return ('A_q', 'B_q',
                'A_k', 'B_k',
                'A_v', 'B_v',
                'A_proj', 'B_proj',
                'A_fc1', 'B_fc1', 'A_fc2', 'B_fc2')
