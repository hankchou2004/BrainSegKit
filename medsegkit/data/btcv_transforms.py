"""MONAI transform pipeline for the BTCV abdominal CT dataset (13 organ classes).

CT images are stored as int16 HU values.  Soft-tissue window [-175, 250] HU
is used to normalise intensity before training.

Label map (already contiguous — no remapping needed):
    0  Background        7  Stomach
    1  Spleen            8  Aorta
    2  Right Kidney      9  Inferior Vena Cava
    3  Left Kidney      10  Portal / Splenic Vein
    4  Gallbladder      11  Pancreas
    5  Esophagus        12  Right Adrenal Gland
    6  Liver            13  Left Adrenal Gland
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    EnsureTyped,
    SpatialPadd,
)

NUM_CLASSES = 14   # 0 (background) + 13 organs

# Soft-tissue CT window
CT_WIN_MIN = -175.0
CT_WIN_MAX =  250.0


def build_btcv_transforms(
    split:      str,
    patch_size: tuple = (96, 96, 96),
    spacing:    tuple = (1.5, 1.5, 2.0),
) -> Compose:
    """Return MONAI Compose pipeline for BTCV train / val / test.

    Keys expected in each data dict:
        "image" → path to img{id}.nii.gz  (CT, HU int16)
        "label" → path to label{id}.nii.gz (optional for test)
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
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_WIN_MIN, a_max=CT_WIN_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
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
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        ]
        return Compose(base + augment)

    return Compose(base)
