# MedSegKit

> 通用醫學影像分割研究框架 — 整合 MONAI 與 PyTorch Lightning，專為多資料集、多模型架構比較與知識蒸餾實驗設計。

---

## 框架定位

```
MedSegKit（你的研究框架）
      ↓ 使用
MONAI 1.5.2          — 模型架構、資料讀取、評估指標
PyTorch Lightning 2.6 — 訓練迴圈、Trainer、checkpoint
      ↓ 運行於
PyTorch 2.11 + CUDA 12.8 + RTX 5070
```

MedSegKit **不複製或修改** MONAI / Lightning 原始碼，而是將兩者作為底層 library，在其上建立：

- 統一模型切換介面（Model Registry）
- 多資料集支援：腦部 MRI（OASIS-1）/ 腹部 CT（BTCV）
- 知識蒸餾訓練框架（4 種 KD 模式）
- 多模型比較評估流程

---

## 支援的資料集

| 資料集 | 影像模態 | 分割類別數 | DataModule |
|--------|---------|-----------|------------|
| OASIS-1 (FreeSurfer aseg) | MRI T1 | 41（腦區） | `OasisDataModule` |
| BTCV | CT | 14（0 背景 + 13 器官） | `BTCVDataModule` |

### BTCV 13 器官類別

| 標籤值 | 器官 | 標籤值 | 器官 |
|--------|------|--------|------|
| 0 | Background | 7 | Stomach |
| 1 | Spleen | 8 | Aorta |
| 2 | Right Kidney | 9 | Inferior Vena Cava |
| 3 | Left Kidney | 10 | Portal / Splenic Vein |
| 4 | Gallbladder | 11 | Pancreas |
| 5 | Esophagus | 12 | Right Adrenal Gland |
| 6 | Liver | 13 | Left Adrenal Gland |

---

## 支援的模型

| 名稱 | 呼叫 key | 來源 | 參數量 | 說明 |
|------|---------|------|--------|------|
| UNet | `"unet"` | MONAI | ~19 M | 標準 3D UNet，residual units |
| UNet++ | `"unet_pp"` | MONAI | ~9 M | Nested UNet，密集 skip connections |
| Attention UNet | `"attention_unet"` | MONAI | ~12 M | 加入 attention gate 的 UNet |
| DynUNet | `"dynunet"` | MONAI | ~20 M | nnUNet 架構，支援 deep supervision |
| SwinUNETR | `"swin_unetr"` | MONAI | ~62 M | Swin Transformer encoder + UNet decoder |
| MedNeXt | `"mednext"` | MONAI | ~18 M | ConvNeXt-style 醫學影像分割 |
| SegResNet | `"segresnet"` | MONAI | ~5 M | ResNet-style encoder-decoder |
| UNETR | `"unetr"` | MONAI | ~93 M | ViT encoder + UNet decoder |
| **MedSAM** | `"medsam"` | 自製 | 90 M（0.16 M trainable） | SAM ViT-B encoder（凍結）+ 輕量 decoder，需下載 checkpoint |
| **U-Mamba** | `"umamba"` | 自製 | ~20 M | Mamba SSM bottleneck，需安裝 `mamba-ssm` |

> **MedSAM 參數說明**：ViT-B encoder（89.7 M）預設凍結，僅訓練 decoder（0.16 M），為參數高效 fine-tuning 策略。

---

## 知識蒸餾（KD）框架

支援 4 種 KD 模式，透過 config YAML 的 `kd.type` 切換：

```
Teacher（大模型，凍結）
    │
    ├── response    → KL divergence（soft label + temperature scaling）
    ├── feature     → L2 / L1 / Attention Transfer（中間層特徵對齊）
    ├── contrastive → InfoNCE / CRD（teacher/student 特徵對比）
    └── combined    → response + feature 同時啟用
    ↓
Student（小模型，訓練）
```

---

## 資料集結構

### OASIS-1 FreeSurfer 41 類腦區分割

**原始資料（轉換前）：**

```
brain_data/
└── oasis1_freesurfer/
    └── {subject}/mri/
        ├── T1.mgz      ← T1 加權影像
        └── aseg.mgz    ← FreeSurfer 自動腦區分割
```

**轉換後（nnUNet 格式）：**

```
dataset/
└── oasis1_freesurfer/
    ├── splits.json          ← train 255 / val 85 / test 85
    ├── imagesTr/            ← 340 個 T1 .nii.gz（train + val）
    ├── labelsTr/            ← 340 個 aseg .nii.gz（train + val）
    ├── imagesTs/            ←  85 個 T1 .nii.gz（test）
    └── labelsTs/            ←  85 個 aseg .nii.gz（test）
```

**FreeSurfer aseg 標籤重映射（非連續 → 0–40）：**

| 重映射後 | 原始值 | 結構名稱 |
|---------|--------|---------|
| 0 | 0, 255 | Background / Unknown |
| 1 | 2 | Left Cerebral White Matter |
| 2 | 3 | Left Cerebral Cortex |
| 3 | 4 | Left Lateral Ventricle |
| ... | ... | ... |
| 20 | 41 | Right Cerebral White Matter |
| ... | ... | ... |
| 40 | 85 | Optic Chiasm |

完整對照見 `medsegkit/data/transforms.py` 的 `FS_LABEL_SRC / FS_LABEL_DST`。

---

### BTCV 腹部 CT 13 器官分割

```
dataset/
└── btcv/
    ├── splits.json          ← train 24 / val 6 / test 20
    ├── Training/
    │   ├── img/             ← 30 個 CT .nii.gz（img{id}.nii.gz）
    │   └── label/           ← 30 個標籤 .nii.gz（label{id}.nii.gz）
    └── Testing/
        └── img/             ← 20 個 CT .nii.gz（無標籤）
```

- 影像：int16 HU 值，512×512×~150 voxels
- 預處理：HU 視窗 `[-175, 250]`（軟組織窗），重採樣至 1.5×1.5×2.0 mm
- 標籤已連續（0–13），訓練時不需重映射

---

## 快速開始

### 1. 安裝環境

```bash
conda create -n medseg python=3.10 -y
conda activate medseg

# PyTorch 2.11 + CUDA 12.8（RTX 5070 / sm_120）
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install "monai[all]==1.5.2" "lightning==2.6.5"
pip install nibabel SimpleITK scipy scikit-learn einops timm wandb

# 安裝 MedSegKit（editable）
pip install -e /path/to/MedSegKit

# 可選：U-Mamba
pip install mamba-ssm causal-conv1d
```

### 2. 轉換 OASIS-1 資料集

```bash
conda activate medseg
python scripts/convert_dataset.py
# 執行後 brain_data/ 自動刪除，保留 dataset/oasis1_freesurfer/
```

### 3. 安裝 MedSAM（可選）

若要使用 `medsam` 模型，需額外安裝 `segment-anything`：

```bash
pip install "medsegkit[sam]"
# 或手動安裝：
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**下載 checkpoint：**

| 權重 | 說明 | 大小 |
|------|------|------|
| `medsam_vit_b.pth` | MedSAM 醫學影像 fine-tuned（推薦） | ~375 MB |
| `sam_vit_b_01ec64.pth` | 原始 SAM ViT-B（Meta） | ~375 MB |

- **MedSAM**（推薦）：[Google Drive](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) → `medsam_vit_b.pth`
- **SAM ViT-B**（原始）：`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

下載後，在 `configs/btcv_medsam.yaml` 中取消 `checkpoint` 行的註解並填入路徑。

### 4. 訓練

```bash
cd /path/to/MedSegKit
conda activate brain_segmention

# OASIS-1 腦部分割（41 類）
python experiments/train.py --config configs/unet.yaml
python experiments/train.py --config configs/dynunet.yaml
python experiments/train.py --config configs/mednext.yaml

# BTCV 腹部 CT 分割（14 類）
python experiments/train.py --config configs/btcv_unet.yaml
python experiments/train.py --config configs/btcv_dynunet.yaml

# BTCV + MedSAM（需先下載 checkpoint，見步驟 3）
python experiments/train.py --config configs/btcv_medsam.yaml

# 知識蒸餾
python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
```

訓練過程會自動記錄以下指標至 TensorBoard（預設），用 `tensorboard --logdir logs/` 檢視：

| 指標 | 說明 |
|------|------|
| `train/loss` | 訓練損失（每 step + 每 epoch） |
| `train/dice` | 訓練 Dice（patch 級別，每 epoch 彙整） |
| `val/loss` | 驗證損失（用於偵測 overfitting） |
| `val/dice` | 驗證 Dice（sliding-window，完整體積） |
| `val/hd95` | 驗證 Hausdorff Distance 95% |
| `lr` | Learning rate（CosineAnnealing） |

若要改用 wandb，在 config 加 `experiment.logger: wandb`。  
Checkpoint 存於 `logs/{experiment.name}/best_model.ckpt`（val/dice 最高時儲存）。

### 5. 多模型比較評估

```bash
# OASIS-1：使用 test split（85 筆，有標籤）
python experiments/evaluate.py \
    --config configs/unet.yaml \
    --ckpts unet:logs/unet_oasis1/best_model.ckpt \
            dynunet:logs/dynunet_oasis1/best_model.ckpt \
            mednext:logs/mednext_oasis1/best_model.ckpt

# BTCV：自動改用 val split（6 筆有標籤；test split 20 筆無標籤）
python experiments/evaluate.py \
    --config configs/btcv_dynunet.yaml \
    --ckpts dynunet:logs/dynunet_btcv/best_model.ckpt \
            medsam:logs/medsam_btcv/best_model.ckpt

# 加上 --per-class 印出各類別 Dice 明細
python experiments/evaluate.py \
    --config configs/btcv_dynunet.yaml \
    --ckpts dynunet:logs/dynunet_btcv/best_model.ckpt \
    --per-class
```

### 6. 推論（儲存預測結果）

對無標籤資料（如 BTCV test split）或任意影像輸出分割遮罩 `.nii.gz`：

```bash
# BTCV test split（20 筆，無標籤，只能做推論）
python experiments/inference.py \
    --config configs/btcv_dynunet.yaml \
    --ckpt   logs/dynunet_btcv/best_model.ckpt \
    --split  test \
    --out    ./predictions/btcv_test

# 單一影像
python experiments/inference.py \
    --config configs/btcv_dynunet.yaml \
    --ckpt   logs/dynunet_btcv/best_model.ckpt \
    --image  /path/to/img0061.nii.gz \
    --out    ./predictions

# OASIS-1 val split
python experiments/inference.py \
    --config configs/unet.yaml \
    --ckpt   logs/unet_oasis1/best_model.ckpt \
    --split  val \
    --out    ./predictions/oasis1_val
```

輸出命名：`{原始檔名}_pred.nii.gz`，保留重採樣後的 affine 矩陣。

### 7. 影像視覺化

```bash
# 互動視窗（需本機 display / WSLg）
python scripts/viewer.py --dataset oasis1 --split train
python scripts/viewer.py --dataset btcv   --split val

# SSH 無頭模式：存 PNG（不需要 display）
python scripts/viewer.py --dataset oasis1 --save --case 0 --save-dir ./viewer_out
python scripts/viewer.py --dataset btcv   --save --case 3 --save-dir ./viewer_out
```

視窗快捷鍵：

| 按鍵 | 功能 |
|------|------|
| `← / →` | 切換 case |
| `A / N` | 全部 labels 開 / 關 |
| `Click`（panel） | 切換單一 label |
| `Scroll` | 縮放 |

OASIS-1 快捷鍵：

| 按鍵 | 功能 |
|------|------|
| `H` | Hippocampus（左右） |
| `V` | Ventricles |
| `C` | Cortex（左右） |
| `1` | 只顯示 White Matter |
| `2` | 只顯示 Gray Matter |
| `3` | 只顯示 CSF / Ventricles |

BTCV 快捷鍵：

| 按鍵 | 功能 |
|------|------|
| `L` | Liver |
| `S` | Spleen |
| `K` | Kidneys（左右） |
| `P` | Pancreas |
| `A` | Aorta |
| `1` | Solid organs（Spleen/Kidneys/Liver） |
| `2` | Vascular（Aorta/IVC/Portal Vein） |
| `3` | GI tract（Esophagus/Stomach/Gallbladder） |

### Python API

```python
import medsegkit

print(medsegkit.list_models())
# ['attention_unet', 'dynunet', 'medsam', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# OASIS-1 腦部分割（41 類）
model = medsegkit.build_model("dynunet", in_channels=1, out_channels=41)

# BTCV 腹部分割（14 類）
model = medsegkit.build_model("dynunet", in_channels=1, out_channels=14)

# MedSAM：SAM ViT-B encoder（凍結）+ 輕量 decoder
model = medsegkit.build_model(
    "medsam",
    in_channels=1,
    out_channels=14,
    checkpoint="/path/to/medsam_vit_b.pth",  # 可選，None 使用隨機初始化
    freeze_encoder=True,
    slice_axis=2,   # 0=sagittal, 1=coronal, 2=axial
    slice_batch=8,  # 每次編碼的 slice 數（OOM 時調低）
)

# DataModules
from medsegkit.data.oasis_module import OasisDataModule
from medsegkit.data.btcv_module import BTCVDataModule

dm_brain = OasisDataModule(dataset_root="/path/to/dataset")
dm_btcv  = BTCVDataModule(dataset_root="/path/to/dataset")
```

---

## 專案結構

```
MedSegKit/
├── medsegkit/
│   ├── models/
│   │   ├── registry.py           # @register_model 裝飾器
│   │   ├── wrappers.py           # MONAI 模型統一包裝（8 個）
│   │   ├── medsam.py             # MedSAM：SAM ViT-B encoder + 輕量 decoder
│   │   └── umamba/umamba.py      # U-Mamba 自製實作
│   ├── losses/
│   │   ├── seg_losses.py         # 分割損失工廠
│   │   └── kd/
│   │       ├── response_kd.py    # Hinton soft-label KD
│   │       ├── feature_kd.py     # 特徵層對齊 KD
│   │       └── contrastive_kd.py # InfoNCE / CRD
│   ├── data/
│   │   ├── transforms.py         # OASIS-1 前處理 + FreeSurfer label remapping
│   │   ├── oasis_module.py       # OASIS-1 LightningDataModule（41 類腦區）
│   │   ├── btcv_transforms.py    # BTCV CT 前處理（HU 視窗化）
│   │   └── btcv_module.py        # BTCV LightningDataModule（14 類腹部器官）
│   ├── engine/
│   │   ├── seg_module.py         # 一般分割 LightningModule
│   │   └── kd_module.py          # 知識蒸餾 LightningModule
│   └── evaluation/
│       └── metrics.py            # Dice / HD95 / NSD 多模型比較
├── configs/
│   ├── unet.yaml                 # OASIS-1 UNet
│   ├── unet_pp.yaml              # OASIS-1 UNet++
│   ├── dynunet.yaml              # OASIS-1 DynUNet
│   ├── mednext.yaml              # OASIS-1 MedNeXt
│   ├── btcv_unet.yaml            # BTCV UNet
│   ├── btcv_dynunet.yaml         # BTCV DynUNet
│   ├── btcv_medsam.yaml          # BTCV MedSAM（SAM ViT-B encoder）
│   └── kd/dynunet_to_unet.yaml   # 知識蒸餾
├── experiments/
│   ├── train.py                  # 訓練入口（依 data.dataset 自動選 DataModule）
│   ├── evaluate.py               # 評估入口（BTCV 自動改用 val split）
│   └── inference.py              # 推論入口（輸出 .nii.gz，支援無標籤 test split）
├── scripts/
│   ├── convert_dataset.py        # .mgz → nnUNet .nii.gz 轉換（OASIS-1）
│   └── viewer.py                 # 通用影像檢視器（OASIS-1 / BTCV）
├── pyproject.toml
└── CLAUDE.md / SKILLS.md         # 每個子目錄均含：Claude Code 上下文 + 技能文件
```

---

## 依賴版本

| 套件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| MONAI | 1.5.2 |
| Lightning | 2.6.5 |
| nibabel | ≥5.0 |
| einops | ≥0.6 |
| timm | ≥0.9 |
| wandb | latest |
| segment-anything | latest（`medsam` 模型需要） |
| mamba-ssm | ≥2.0（`umamba` 模型需要） |

---

## 參考文獻

- **UNet**: Ronneberger et al., MICCAI 2015
- **UNet++**: Zhou et al., MICCAI Workshop 2018
- **nnUNet / DynUNet**: Isensee et al., Nature Methods 2021
- **SwinUNETR**: Tang et al., CVPR 2022
- **MedNeXt**: Roy et al., MICCAI 2023
- **U-Mamba**: Ma et al., ArXiv 2024
- **KD (Response)**: Hinton et al., NeurIPS Workshop 2015
- **KD (Feature / FitNets)**: Romero et al., ICLR 2015
- **KD (Contrastive / CRD)**: Tian et al., ICLR 2020
- **OASIS-1**: Marcus et al., Journal of Cognitive Neuroscience 2007
- **BTCV**: Landman et al., MICCAI Workshop 2015
- **SAM**: Kirillov et al., ICCV 2023
- **MedSAM**: Ma et al., Nature Communications 2024
