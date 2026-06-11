"""MONAI transform pipeline for OASIS-1 brain MRI (FreeSurfer aseg labels).

FreeSurfer aseg: 41 non-contiguous labels → remapped to 0–40.
Label 255 (Unknown) is remapped to 0 (background).
"""

from __future__ import annotations

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
# FreeSurfer aseg: 41 unique labels (verified from OASIS-1 scans)
# 255 = FreeSurfer "Unknown" → mapped to 0 (background)
# ---------------------------------------------------------------------------
FS_LABEL_SRC = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 26, 28, 30, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53,
    54, 58, 60, 62, 72, 78, 79, 81, 82, 85, 255,
]
FS_LABEL_DST = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 0,   # 255 → 0
]

NUM_CLASSES = 41   # 0 (bg) + 40 structures


def build_transforms(
    split:      str,
    patch_size: tuple = (128, 128, 128),
    spacing:    tuple = (1.0, 1.0, 1.0),
) -> Compose:
    """Return a MONAI Compose pipeline for train / val / test.

    Keys expected in each data dict:
        "image" → path to *_0000.nii.gz  (T1)
        "label" → path to *.nii.gz       (aseg)
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
        MapLabelValued(
            keys=["label"],
            orig_labels=FS_LABEL_SRC,
            target_labels=FS_LABEL_DST,
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99,
            b_min=0.0, b_max=1.0, clip=True,
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
                roi_size=patch_size, random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(base + augment)

    return Compose(base)
