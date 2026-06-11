"""MONAI transform pipelines for OASIS1 brain MRI.

FreeSurfer aseg.mgz uses non-contiguous integer labels (e.g. 2,3,4,7,8...).
MapLabelValued remaps them to contiguous 0-35 before training.
"""

from __future__ import annotations

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    MapLabelValued,
    EnsureTyped,
    SpatialPadd,
)

# ---------------------------------------------------------------------------
# Standard FreeSurfer aseg label → contiguous index mapping
# 35 structures + background = 36 classes
# ---------------------------------------------------------------------------
ASEG_SRC = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54,
    58, 60, 77, 85,
]
ASEG_DST = list(range(len(ASEG_SRC)))   # 0 … 35
NUM_CLASSES = len(ASEG_SRC)             # 36


def build_transforms(
    split:      str,
    patch_size: tuple = (128, 128, 128),
    spacing:    tuple = (1.0, 1.0, 1.0),
) -> Compose:
    """Return a MONAI Compose pipeline for train / val / test.

    Keys expected in each data dict:
        "image" → path to T1.mgz
        "label" → path to aseg.mgz
    """
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        # Remap aseg labels to 0-35
        MapLabelValued(
            keys=["label"],
            orig_labels=ASEG_SRC,
            target_labels=ASEG_DST,
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        EnsureTyped(keys=["image", "label"]),
    ]

    if split == "train":
        augment = [
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=patch_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(base + augment)

    # val / test: centre-crop only (full volume used with sliding-window inferer)
    return Compose(base)
