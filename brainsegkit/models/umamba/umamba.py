"""U-Mamba: UNet encoder-decoder with Mamba SSM bottleneck blocks.

Architecture follows U-Mamba (Ma et al., 2024):
  CNN encoder → Mamba bottleneck → CNN decoder with skip connections
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brainsegkit.models.registry import register_model
from brainsegkit.models.umamba.mamba_block import MambaBlock3D


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


@register_model("umamba")
class UMamba3D(nn.Module):
    """U-Mamba for 3-D volumetric segmentation.

    Encoder: 4× CNN downsampling blocks
    Bottleneck: Mamba SSM blocks (sequence modelling)
    Decoder: 4× CNN upsampling blocks with skip connections
    """

    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 36,
        features:     tuple = (32, 64, 128, 256),
        mamba_depth:  int = 2,
        d_state:      int = 16,
    ):
        super().__init__()
        f = features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1], stride=2)
        self.enc3 = ConvBlock(f[1], f[2], stride=2)
        self.enc4 = ConvBlock(f[2], f[3], stride=2)

        # Bottleneck — Mamba SSM blocks
        self.bottleneck = nn.Sequential(
            *[MambaBlock3D(f[3], d_state=d_state) for _ in range(mamba_depth)]
        )

        # Decoder
        self.dec4 = UpBlock(f[3], f[2], f[2])
        self.dec3 = UpBlock(f[2], f[1], f[1])
        self.dec2 = UpBlock(f[1], f[0], f[0])
        self.head  = nn.Conv3d(f[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d = self.dec4(b,  e3)
        d = self.dec3(d,  e2)
        d = self.dec2(d,  e1)
        return self.head(d)
