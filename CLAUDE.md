# MedSegKit — Claude Code 上下文

## 專案定位 / Project Overview

MedSegKit 是基於 MONAI + PyTorch Lightning 的醫學影像分割研究框架，支援多資料集、多模型切換與知識蒸餾。

A medical image segmentation research framework built on MONAI + PyTorch Lightning, supporting multi-dataset, multi-model switching, and knowledge distillation.

## 環境 / Environment

- conda env: `brain_segmention`（啟動：`conda activate brain_segmention`）
- Python 3.10, PyTorch 2.11+cu128, MONAI 1.5.2, Lightning 2.6.5
- GPU: RTX 5070（sm_120，CUDA 12.8）
- WSL2/WSLg：viewer.py 需 `DISPLAY=:0`，SSH 無頭模式用 `--save`

## 主要進入點 / Entry Points

```bash
cd /home/hank/medical_segmention/MedSegKit
conda activate brain_segmention

python experiments/train.py     --config configs/btcv_unet.yaml
python experiments/evaluate.py  --config configs/btcv_unet.yaml --ckpts unet:logs/.../best_model.ckpt
python experiments/inference.py --config configs/btcv_unet.yaml --ckpt logs/.../best_model.ckpt --split test --out ./predictions
python scripts/viewer.py        --dataset btcv --split val
```

## 資料集路徑 / Dataset Paths

- OASIS-1: `/home/hank/medical_segmention/dataset/oasis1_freesurfer/`
- BTCV:    `/home/hank/medical_segmention/dataset/btcv/`

## 重要慣例 / Key Conventions

- **新增模型**：在 `medsegkit/models/` 建立檔案，用 `@register_model("key")` 裝飾，並在 `medsegkit/models/__init__.py` import
- **新增資料集**：新增 `medsegkit/data/{name}_module.py`（LightningDataModule）+ `{name}_transforms.py`，在 `experiments/train.py` 的 `build_datamodule()` 加入路由
- **Config YAML**：必須包含 `model.name`（對應 registry key）、`data.dataset`（`oasis1` 或 `btcv`）、`training.max_epochs`
- BTCV test split 無標籤，evaluate.py 自動改用 val split；inference.py 可直接跑 test split

## 禁止事項 / Do NOT

- 不要修改 MONAI / Lightning 原始碼
- 不要在 `medsegkit/models/wrappers.py` 以外的地方直接 import MONAI model class（統一走 registry）
- BTCV test 的 20 筆資料沒有 label，不要嘗試計算 Dice
