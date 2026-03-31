import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 1024        # max sequence length
    vocab_size: int = 50304       # 50257 padded to nearest multiple of 64
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
    dropout:    float = 0.0
    bias:       bool = True       # False is slightly faster & better


# ── Modules ───────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch's built-in forces bias=True)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
