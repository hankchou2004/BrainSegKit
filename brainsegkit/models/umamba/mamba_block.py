"""Mamba SSM block for volumetric (3-D) feature maps.

Requires mamba-ssm and causal-conv1d:
    pip install mamba-ssm causal-conv1d

Reference: U-Mamba (Ma et al., 2024) https://arxiv.org/abs/2401.04722
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class MambaBlock3D(nn.Module):
    """Wraps a 1-D Mamba SSM for 3-D volumetric feature maps.

    Flattens (B, C, H, W, D) → (B, H*W*D, C), applies Mamba, reshapes back.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm not installed. Run: pip install mamba-ssm causal-conv1d\n"
                "Note: RTX 50xx (sm_120) may require building from source."
            )
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        # (B, C, H, W, D) → (B, seq, C)
        seq = rearrange(x, "b c h w d -> b (h w d) c")
        seq = seq + self.mamba(self.norm(seq))
        return rearrange(seq, "b (h w d) c -> b c h w d", h=H, w=W, d=D)
