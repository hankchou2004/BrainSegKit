# medsegkit/data/ — Claude Code 上下文

## 用途 / Purpose

資料讀取、前處理、DataModule。每個資料集有獨立的 DataModule（Lightning）和 transform pipeline（MONAI Compose）。

Data loading, preprocessing, and DataModules. Each dataset has its own DataModule (Lightning) and transform pipeline (MONAI Compose).

## 檔案說明 / Files

| 檔案 | 用途 |
|------|------|
| `transforms.py` | OASIS-1 前處理 + FreeSurfer aseg label 重映射（非連續 → 0–40） |
| `oasis_module.py` | `OasisDataModule`：OASIS-1 腦部 MRI，41 類 |
| `btcv_transforms.py` | BTCV CT 前處理（HU 視窗 [-175, 250]，重採樣 1.5×1.5×2.0 mm） |
| `btcv_module.py` | `BTCVDataModule`：BTCV 腹部 CT，14 類（0 背景+13 器官） |

## OASIS-1 Label 重映射 / OASIS-1 Label Remapping

FreeSurfer aseg 標籤為非連續值（0, 2, 3, 4... 85），必須重映射為 0–40。
重映射表：`transforms.py` 的 `FS_LABEL_SRC`（原始值）和 `FS_LABEL_DST`（目標值）。
使用 MONAI `MapLabelValued` 執行。

## BTCV 前處理 / BTCV Preprocessing

- HU 值：int16，視窗 [-175, 250] → 正規化至 [0, 1]（`ScaleIntensityRanged`）
- 重採樣：`Spacingd`，pixdim=(1.5, 1.5, 2.0)，label 用 nearest
- 方向：`Orientationd`，axcodes="RAS"
- 標籤已連續（0–13），不需重映射

## BTCV Test Split 注意 / BTCV Test Split Note

`BTCVDataModule._load_split("test")` 中 `"label"` 欄位設為 image 路徑（佔位符）。  
推論時忽略此 label；不要用 test split 計算 Dice。

## 新增資料集 / Adding a New Dataset

1. 建立 `{name}_transforms.py`，定義 `build_{name}_transforms(split, patch_size, spacing)`
2. 建立 `{name}_module.py`，實作 `LightningDataModule`，`_load_split()` 回傳 `list[dict]`（含 "image", "label" 鍵）
3. 在 `experiments/train.py` 的 `build_datamodule()` 加入路由
