"""Wraps MONAI architectures into the MedSegKit registry with OASIS1 defaults.

All wrappers expose a unified signature:
    __init__(in_channels, out_channels, **extra)
    forward(x) -> logits  shape (B, C, H, W, D)
"""

from __future__ import annotations

from monai.networks.nets import (
    UNet,
    BasicUNetPlusPlus,
    AttentionUnet,
    DynUNet,
    SwinUNETR,
    MedNeXt,
    SegResNet,
    UNETR,
)

from medsegkit.models.registry import register_model

# ---------------------------------------------------------------------------
# Default channel/stride settings for 3-D brain MRI at ~1 mm isotropic
# ---------------------------------------------------------------------------
_UNET_CHANNELS   = (32, 64, 128, 256, 512)
_UNET_STRIDES    = (2, 2, 2, 2)

_DYN_KERNELS = [[3,3,3]] * 6
_DYN_STRIDES = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
_DYN_UP_KS   = [[2,2,2]] * 5


@register_model("unet")
class UNet3D(UNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=kw.pop("channels", _UNET_CHANNELS),
            strides=kw.pop("strides", _UNET_STRIDES),
            num_res_units=kw.pop("num_res_units", 2),
            **kw,
        )


@register_model("unet_pp")
class UNetPP3D(BasicUNetPlusPlus):
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=kw.pop("features", (32, 32, 64, 128, 256, 32)),
            **kw,
        )


@register_model("attention_unet")
class AttentionUNet3D(AttentionUnet):
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=kw.pop("channels", (32, 64, 128, 256)),
            strides=kw.pop("strides", (2, 2, 2)),
            **kw,
        )


@register_model("dynunet")
class DynUNet3D(DynUNet):
    """nnUNet-style dynamic UNet (MONAI implementation).

    deep_supervision=True activates auxiliary heads used during training;
    only the first output (full-res) is used at inference.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kw.pop("kernel_size", _DYN_KERNELS),
            strides=kw.pop("strides", _DYN_STRIDES),
            upsample_kernel_size=kw.pop("upsample_kernel_size", _DYN_UP_KS),
            deep_supervision=kw.pop("deep_supervision", True),
            res_block=kw.pop("res_block", True),
            **kw,
        )


@register_model("swin_unetr")
class SwinUNETR3D(SwinUNETR):
    def __init__(self, in_channels: int = 1, out_channels: int = 36,
                 img_size: tuple = (128, 128, 128), **kw):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kw.pop("feature_size", 48),
            use_checkpoint=kw.pop("use_checkpoint", True),
            **kw,
        )


@register_model("mednext")
class MedNeXt3D(MedNeXt):
    """MedNeXt — ConvNeXt-style encoder/decoder (MONAI 1.6+)."""
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            enc_num_block=kw.pop("enc_num_block", [2, 2, 2, 2]),
            dec_num_block=kw.pop("dec_num_block", [2, 2, 2, 2]),
            **kw,
        )


@register_model("segresnet")
class SegResNet3D(SegResNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 36, **kw):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=kw.pop("init_filters", 32),
            **kw,
        )


@register_model("unetr")
class UNETR3D(UNETR):
    def __init__(self, in_channels: int = 1, out_channels: int = 36,
                 img_size: tuple = (128, 128, 128), **kw):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=kw.pop("feature_size", 16),
            hidden_size=kw.pop("hidden_size", 768),
            mlp_dim=kw.pop("mlp_dim", 3072),
            num_heads=kw.pop("num_heads", 12),
            spatial_dims=3,
            **kw,
        )
