"""MedSAM 3D segmentation model for MedSegKit.

Architecture
------------
* **Image encoder**: SAM ViT-B (frozen by default), fine-tuned on medical images
  when a MedSAM checkpoint is provided.
* **Decoder**: lightweight 2-layer transpose-conv head that upsamples the
  64×64 ViT feature map back to the slice resolution and produces
  multi-class logits.
* **3D strategy**: each axial (or sagittal / coronal) 2-D slice is encoded
  independently; predictions are stacked back into a 3-D volume.

Usage
-----
    python experiments/train.py --config configs/btcv_medsam.yaml

Config snippet::

    model:
      name: medsam
      in_channels: 1
      out_channels: 14
      checkpoint: /path/to/medsam_vit_b.pth   # or sam_vit_b_01ec64.pth
      freeze_encoder: true
      slice_axis: 2                            # 0=sag 1=cor 2=axi

Checkpoint download
-------------------
* MedSAM weights (recommended):
    https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
    → medsam_vit_b.pth  (~375 MB)

* Original SAM ViT-B (Meta):
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from medsegkit.models.registry import register_model

# SAM ViT-B constants
_SAM_INPUT_SIZE = 1024   # expected input resolution
_SAM_FEAT_SIZE  =   64   # neck output spatial size
_SAM_EMBED_DIM  =  256   # neck output channels


# ── lightweight 2-D decoder ───────────────────────────────────────────────────
class _Decoder(nn.Module):
    """64×64 feature map → multi-class 2-D logits."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 2, stride=2),   # 128×128
            nn.InstanceNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),      # 256×256
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, out_ch, 1),                       # class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ── MedSAM 3D wrapper ─────────────────────────────────────────────────────────
@register_model("medsam")
class MedSAM3D(nn.Module):
    """MedSAM-based 3-D segmentation model.

    Processes 3-D volumes slice-by-slice using a SAM ViT-B image encoder
    (initialised from MedSAM or original SAM weights), then aggregates
    2-D predictions into a 3-D segmentation map.

    The encoder is frozen by default (parameter-efficient fine-tuning):
    only the ~3 M-parameter decoder is trained.

    Args:
        in_channels:    Input channels (1 for grayscale MRI/CT).
        out_channels:   Number of segmentation classes.
        checkpoint:     Path to ``medsam_vit_b.pth`` or ``sam_vit_b_*.pth``.
                        Pass ``None`` to use random ViT weights (not recommended).
        freeze_encoder: Whether to freeze the ViT encoder (default ``True``).
        slice_axis:     Axis along which slices are extracted (0=sagittal,
                        1=coronal, 2=axial). Default 2 (axial).
        slice_batch:    Number of slices to encode in one GPU call.
                        Reduce if GPU memory is tight (default 16).
    """

    def __init__(
        self,
        in_channels:    int  = 1,
        out_channels:   int  = 14,
        checkpoint:     str  = None,
        freeze_encoder: bool = True,
        slice_axis:     int  = 2,
        slice_batch:    int  = 16,
        **kw,
    ):
        super().__init__()

        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "MedSAM requires the segment-anything package.\n"
                "Install: pip install git+https://github.com/facebookresearch/"
                "segment-anything.git"
            ) from exc

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.image_encoder = sam.image_encoder   # ViT-B neck output: (B,256,64,64)

        if freeze_encoder:
            self.image_encoder.requires_grad_(False)

        self.decoder      = _Decoder(_SAM_EMBED_DIM, out_channels)
        self.out_channels = out_channels
        self.slice_axis   = slice_axis
        self.slice_batch  = slice_batch

    # ── internal helpers ──────────────────────────────────────────────────────
    def _to_sam_input(self, x2d: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) float → (B,3,1024,1024), normalised to [0,1]."""
        x3 = x2d.repeat(1, 3, 1, 1)
        x3 = F.interpolate(x3, (_SAM_INPUT_SIZE, _SAM_INPUT_SIZE),
                           mode="bilinear", align_corners=False)
        # per-sample min-max normalisation
        b = x3.shape[0]
        mn = x3.view(b, -1).min(1).values.view(b, 1, 1, 1)
        mx = x3.view(b, -1).max(1).values.view(b, 1, 1, 1)
        return (x3 - mn) / (mx - mn + 1e-8)

    def _encode_batch(self, slices: torch.Tensor) -> torch.Tensor:
        """slices: (N,1,H,W) → logits: (N,C,H,W)."""
        H, W = slices.shape[2:]
        sam_in = self._to_sam_input(slices)

        # encoder: frozen → no grad; unfrozen → grad enabled
        enc_ctx = torch.no_grad() if not next(
            self.image_encoder.parameters()).requires_grad else torch.enable_grad()
        with enc_ctx:
            feats = self.image_encoder(sam_in)     # (N,256,64,64)

        logits = self.decoder(feats)               # (N,C,256,256)
        return F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 1, H, W, D)`` float tensor.
        Returns:
            logits: ``(B, C, H, W, D)`` float tensor.
        """
        B, _, H, W, D = x.shape
        ax = self.slice_axis

        # permute so the slice dimension comes last: (B,1,s0,s1,n_slices)
        # ax=0 → (B,1,W,D,H); ax=1 → (B,1,H,D,W); ax=2 → (B,1,H,W,D)
        perm = {0: (0,1,2,3,4), 1: (0,1,2,3,4), 2: (0,1,2,3,4)}
        if ax == 0:
            xp = x.permute(0, 1, 2, 3, 4)          # slice over H → last
            # reorder to (B,1,W,D,H)
            xp = x.permute(0, 1, 3, 4, 2)
            s0, s1, n = W, D, H
        elif ax == 1:
            xp = x.permute(0, 1, 2, 4, 3)           # (B,1,H,D,W)
            s0, s1, n = H, D, W
        else:
            xp = x                                   # (B,1,H,W,D)
            s0, s1, n = H, W, D

        # reshape: (B, 1, s0, s1, n) → (B*n, 1, s0, s1)
        xp  = xp.permute(0, 4, 1, 2, 3).reshape(B * n, 1, s0, s1)

        # chunked encoding (controls GPU peak memory)
        out_slices = []
        for start in range(0, B * n, self.slice_batch):
            chunk = xp[start : start + self.slice_batch]
            out_slices.append(self._encode_batch(chunk))    # (chunk,C,s0,s1)

        out = torch.cat(out_slices, dim=0)                  # (B*n, C, s0, s1)
        out = out.reshape(B, n, self.out_channels, s0, s1)  # (B,n,C,s0,s1)
        out = out.permute(0, 2, 3, 4, 1)                    # (B,C,s0,s1,n)

        # restore original axis order
        if ax == 0:
            out = out.permute(0, 1, 4, 2, 3)   # (B,C,H,W,D)
        elif ax == 1:
            out = out.permute(0, 1, 2, 4, 3)   # (B,C,H,W,D)
        # ax==2: already (B,C,H,W,D)

        return out
