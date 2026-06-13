# MedSegKit — 技能總覽 / Skills Overview

## 可以做什麼 / What This Framework Can Do

### 訓練 / Training
- 10 種模型（UNet / UNet++ / AttentionUNet / DynUNet / SwinUNETR / MedNeXt / SegResNet / UNETR / MedSAM / U-Mamba）
- 2 個資料集（OASIS-1 腦部 MRI 41 類 / BTCV 腹部 CT 14 類）
- 4 種知識蒸餾模式（response / feature / contrastive / combined）
- Sliding-window inference、Dice+HD95 自動評估、EarlyStopping、ModelCheckpoint

### 評估 / Evaluation
- 多模型 Dice / HD95 比較表（`experiments/evaluate.py`）
- OASIS-1 使用 test split；BTCV 因 test 無標籤，自動改用 val split

### 推論 / Inference
- 批次儲存預測結果為 `.nii.gz`（`experiments/inference.py`）
- 支援 dataset split 或單一影像檔輸入
- 輸出保留重採樣後的 affine 矩陣

### 視覺化 / Visualization
- 互動式多軸向切片檢視（`scripts/viewer.py`）
- 支援 OASIS-1 / BTCV，label panel 點擊切換
- SSH 無頭模式：`--save` 儲存 PNG

## 子目錄技能 / Sub-directory Skills

| 目錄 | 技能 |
|------|------|
| `medsegkit/models/` | 模型建立、Registry 查詢 |
| `medsegkit/data/` | DataModule、Transform pipeline |
| `medsegkit/engine/` | 訓練/KD Lightning Module |
| `medsegkit/losses/` | 分割損失、知識蒸餾損失 |
| `medsegkit/evaluation/` | Dice / HD95 / NSD 多模型比較 |
| `configs/` | YAML 設定檔模板 |
| `experiments/` | 訓練、評估、推論入口 |
| `scripts/` | 資料轉換、互動視覺化 |

## Python API 快速參考 / Quick API Reference

```python
import medsegkit

# 列出所有可用模型
medsegkit.list_models()
# ['attention_unet', 'dynunet', 'medsam', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# 建立模型
model = medsegkit.build_model("dynunet", in_channels=1, out_channels=14)
model = medsegkit.build_model("medsam",  in_channels=1, out_channels=14,
                               checkpoint="/path/to/medsam_vit_b.pth")

# DataModules
from medsegkit.data.oasis_module import OasisDataModule
from medsegkit.data.btcv_module  import BTCVDataModule

dm = BTCVDataModule(dataset_root="/home/hank/medical_segmention/dataset")
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))  # {"image": ..., "label": ...}
```
