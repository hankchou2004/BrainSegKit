# medsegkit/data/ — 技能 / Skills

## DataModule 使用 / DataModule Usage

```python
from medsegkit.data.oasis_module import OasisDataModule
from medsegkit.data.btcv_module  import BTCVDataModule

# OASIS-1
dm = OasisDataModule(
    dataset_root="/home/hank/medical_segmention/dataset",
    patch_size=(128, 128, 128),
    batch_size=2,
)
dm.setup("fit")
train_loader = dm.train_dataloader()  # batch: {"image": (B,1,H,W,D), "label": (B,1,H,W,D)}
val_loader   = dm.val_dataloader()
dm.setup("test")
test_loader  = dm.test_dataloader()

# BTCV
dm = BTCVDataModule(
    dataset_root="/home/hank/medical_segmention/dataset",
    patch_size=(96, 96, 96),
    spacing=(1.5, 1.5, 2.0),
    batch_size=2,
    cache_rate=0.1,  # 快取 10% 到 RAM
)
```

## Transform Pipeline

```python
from medsegkit.data.transforms       import build_transforms        # OASIS-1
from medsegkit.data.btcv_transforms  import build_btcv_transforms   # BTCV

tf_train = build_transforms("train", patch_size=(128,128,128), spacing=(1.0,1.0,1.0))
tf_val   = build_transforms("val",   patch_size=(128,128,128), spacing=(1.0,1.0,1.0))

tf_btcv  = build_btcv_transforms("train", patch_size=(96,96,96), spacing=(1.5,1.5,2.0))
```

## 類別常數 / Class Constants

```python
from medsegkit.data.transforms      import NUM_CLASSES   # 41（OASIS-1）
from medsegkit.data.btcv_transforms import NUM_CLASSES   # 14（BTCV）
from medsegkit.data.btcv_module     import BTCV_CLASSES  # 14-element list
```

## 資料格式 / Data Format

- 所有影像：NIfTI1 `.nii.gz`，nnUNet 命名規則
- OASIS-1 image：`{subject_id}_0000.nii.gz`；label：`{subject_id}.nii.gz`
- BTCV image：`img{id}.nii.gz`；label：`label{id}.nii.gz`（Training only）
- splits.json：`{"splits": {"train": [...], "val": [...], "test": [...]}}`
