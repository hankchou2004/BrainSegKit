# BrainSegKit

> 腦部 MRI 分割研究框架 — 整合 MONAI 與 PyTorch Lightning，專為多模型比較與知識蒸餾實驗設計。

---

## 框架定位

```
BrainSegKit（你的研究框架）
      ↓ 使用
MONAI 1.5.2          — 模型架構、資料讀取、評估指標
PyTorch Lightning 2.6 — 訓練迴圈、Trainer、checkpoint
      ↓ 運行於
PyTorch 2.11 + CUDA 12.8 + RTX 5070
```

BrainSegKit **不複製或修改** MONAI / Lightning 的原始碼，而是將兩者作為底層 library，在其上建立：
- 統一的模型切換介面（Model Registry）
- 腦部 MRI 專用資料流（OASIS-1）
- 知識蒸餾訓練框架
- 多模型比較評估流程

---

## 與 MONAI / PyTorch Lightning 的關係與差異

### MONAI

| 功能 | MONAI 提供 | BrainSegKit 做了什麼 |
|------|-----------|---------------------|
| 模型架構 | UNet、UNet++、DynUNet、SwinUNETR、MedNeXt 等 | 統一包裝為相同介面，用 `build_model("unet")` 一行呼叫 |
| 資料讀取 | `LoadImaged`、`CacheDataset`、`DataLoader` | 整合 OASIS-1 CSV splits，自動建立 train/val/test |
| 標籤前處理 | `MapLabelValued` | 預設設定 FreeSurfer aseg 35 個腦區的 label remapping |
| 損失函數 | `DiceLoss`、`DiceCELoss`、`FocalLoss` | 新增 `build_seg_loss("dice_ce")` 工廠函數 |
| 評估指標 | `DiceMetric`、`HausdorffDistanceMetric` | 整合 Dice + HD95 + NSD，提供多模型比較表格輸出 |
| 推理 | `SlidingWindowInferer` | 在 val/test step 自動啟用 sliding-window 推理 |
| **不提供** | KD 框架 | **BrainSegKit 自行實作** Response / Feature / Contrastive KD |
| **不提供** | 多架構統一比較流程 | **BrainSegKit 自行實作** Model Registry + evaluate.py |

### PyTorch Lightning

| 功能 | Lightning 提供 | BrainSegKit 做了什麼 |
|------|--------------|---------------------|
| 訓練迴圈 | `Trainer`、`LightningModule` | 繼承為 `SegModule`（一般分割）與 `KDModule`（知識蒸餾） |
| 資料管理 | `LightningDataModule` | 繼承為 `OasisDataModule`，內建 OASIS-1 路徑解析 |
| Checkpoint | `ModelCheckpoint` | 預設監控 `val/dice`，儲存最佳模型 |
| 早停 | `EarlyStopping` | 預設 patience=50 epoch |
| 混合精度 | `precision="16-mixed"` | 預設啟用，節省 VRAM |
| **不提供** | Teacher-student 凍結邏輯 | **BrainSegKit 自行實作**：teacher 自動 `requires_grad=False` |
| **不提供** | YAML config 驅動訓練 | **BrainSegKit 自行實作**：`train.py --config configs/unet.yaml` |

### BrainSegKit 自行實作的部分（兩者都沒有）

| 模組 | 說明 |
|------|------|
| `models/registry.py` | `@register_model` 裝飾器，所有模型統一用 `build_model("名稱")` 建立 |
| `models/umamba/` | U-Mamba 架構（CNN encoder + Mamba SSM bottleneck + CNN decoder） |
| `losses/kd/response_kd.py` | Hinton soft-label KD：CE + KL divergence + temperature scaling |
| `losses/kd/feature_kd.py` | 中間層特徵對齊：L2 / L1 / Attention Transfer |
| `losses/kd/contrastive_kd.py` | InfoNCE-based CRD：teacher/student 特徵對比學習 |
| `engine/kd_module.py` | KDModule：teacher 凍結、4 種 KD 模式、統一訓練介面 |
| `evaluation/metrics.py` | 多模型結果統整，輸出 Dice / HD95 / NSD 比較表格 |

---

## 支援的模型

| 名稱 | 呼叫 key | 來源 | 說明 |
|------|---------|------|------|
| UNet | `"unet"` | MONAI | 標準 3D UNet，residual units |
| UNet++ | `"unet_pp"` | MONAI | Nested UNet，密集 skip connections |
| Attention UNet | `"attention_unet"` | MONAI | 加入 attention gate 的 UNet |
| DynUNet | `"dynunet"` | MONAI | nnUNet 架構，支援 deep supervision |
| SwinUNETR | `"swin_unetr"` | MONAI | Swin Transformer encoder + UNet decoder |
| MedNeXt | `"mednext"` | MONAI | ConvNeXt-style 醫學影像分割 |
| SegResNet | `"segresnet"` | MONAI | ResNet-style encoder-decoder |
| UNETR | `"unetr"` | MONAI | ViT encoder + UNet decoder |
| **U-Mamba** | `"umamba"` | 自製 | Mamba SSM bottleneck，需安裝 `mamba-ssm` |

新增模型只需：
```python
from brainsegkit.models.registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=36, **kw):
        ...
```

---

## 知識蒸餾（KD）框架

支援 4 種 KD 模式，透過 config YAML 的 `kd.type` 切換：

```
Teacher（大模型，凍結）
    │
    ├── response   → KL divergence（soft label + temperature）
    ├── feature    → L2/L1/AT（中間層特徵對齊）
    ├── contrastive→ InfoNCE（teacher/student 特徵對比）
    └── combined   → response + feature 同時啟用
    ↓
Student（小模型，訓練）
```

---

## 資料集：OASIS-1

- **影像**：`brain_data/oasis1_freesurfer/{subject}/mri/T1.mgz`
- **標籤**：`brain_data/oasis1_freesurfer/{subject}/mri/aseg.mgz`（FreeSurfer 35 腦區）
- **分割**：`oasis1_splits.csv`（train 255 / val 85 / test 85）

FreeSurfer aseg 標籤自動 remapping 至連續整數 0–35：
```
0 → 0 (Background)       2 → 1 (L-White-Matter)
3 → 2 (L-Cortex)         4 → 3 (L-Lateral-Ventricle)
... 共 36 個類別
```

---

## 快速開始

### 安裝

```bash
conda create -n brain_segmention python=3.10 -y
conda activate brain_segmention

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install "monai[all]==1.5.2" "lightning==2.6.5"
pip install nibabel SimpleITK scipy scikit-learn einops timm wandb

pip install -e /home/hank/medical_segmention/BrainSegKit

# 可選：U-Mamba（需 RTX 50xx 相容版本）
pip install mamba-ssm causal-conv1d

# 可選：nnUNet 完整 pipeline
pip install nnunetv2
```

### 訓練

```bash
cd /home/hank/medical_segmention/BrainSegKit

# 一般分割訓練
python experiments/train.py --config configs/unet.yaml
python experiments/train.py --config configs/dynunet.yaml
python experiments/train.py --config configs/mednext.yaml

# 知識蒸餾（先訓練 teacher，再蒸餾 student）
python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
```

### 多模型比較評估

```bash
python experiments/evaluate.py \
    --config configs/unet.yaml \
    --ckpts unet:logs/unet_oasis1/best_model.ckpt \
            dynunet:logs/dynunet_oasis1/best_model.ckpt \
            mednext:logs/mednext_oasis1/best_model.ckpt
```

輸出範例：
```
=============================================
Model                Mean Dice  Mean HD95  Mean NSD
=============================================
unet                    0.8312      4.21    0.8541
dynunet                 0.8671      3.87    0.8823
mednext                 0.8794      3.62    0.8951
=============================================
```

### Python API

```python
import brainsegkit

# 列出所有可用模型
print(brainsegkit.list_models())
# ['attention_unet', 'dynunet', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# 建立模型
model = brainsegkit.build_model("dynunet", in_channels=1, out_channels=36)

# 切換模型只需改一個字串
model = brainsegkit.build_model("mednext", in_channels=1, out_channels=36)
```

---

## 專案結構

```
BrainSegKit/
├── brainsegkit/
│   ├── models/
│   │   ├── registry.py          # Model Registry
│   │   ├── wrappers.py          # MONAI 模型統一包裝
│   │   └── umamba/              # U-Mamba 自製實作
│   ├── losses/
│   │   ├── seg_losses.py        # 分割損失工廠
│   │   └── kd/                  # 知識蒸餾損失
│   │       ├── response_kd.py   # Hinton KD
│   │       ├── feature_kd.py    # 特徵對齊
│   │       └── contrastive_kd.py# CRD
│   ├── data/
│   │   ├── transforms.py        # MONAI 前處理 pipeline
│   │   └── oasis_module.py      # Lightning DataModule
│   ├── engine/
│   │   ├── seg_module.py        # 一般分割 LightningModule
│   │   └── kd_module.py         # KD LightningModule
│   └── evaluation/
│       └── metrics.py           # 多模型比較表格
├── configs/                     # YAML 實驗配置
├── experiments/
│   ├── train.py                 # 訓練入口
│   └── evaluate.py              # 評估入口
└── pyproject.toml
```

---

## 依賴版本

| 套件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.11.0+cu128 | 深度學習框架 |
| MONAI | 1.5.2 | 醫學影像模型、transforms、metrics |
| Lightning | 2.6.5 | 訓練迴圈、Trainer |
| nibabel | ≥5.0 | 讀取 .mgz / .nii.gz |
| SimpleITK | ≥2.0 | 醫學影像處理 |
| einops | ≥0.6 | Tensor 維度操作（UMamba） |
| timm | ≥0.9 | Transformer backbone |

---

## 參考文獻

- **UNet**: Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
- **UNet++**: Zhou et al., UNet++: A Nested U-Net Architecture, MICCAI Workshop 2018
- **DynUNet / nnUNet**: Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature Methods 2021
- **SwinUNETR**: Tang et al., Self-supervised pre-training of swin transformers for 3d medical image analysis, CVPR 2022
- **MedNeXt**: Roy et al., MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation, MICCAI 2023
- **U-Mamba**: Ma et al., U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation, ArXiv 2024
- **KD (Response)**: Hinton et al., Distilling the Knowledge in a Neural Network, NeurIPS Workshop 2015
- **KD (Feature)**: Romero et al., FitNets: Hints for Thin Deep Nets, ICLR 2015
- **KD (Contrastive)**: Tian et al., Contrastive Representation Distillation, ICLR 2020
- **OASIS-1**: Marcus et al., Open Access Series of Imaging Studies, Journal of Cognitive Neuroscience 2007
