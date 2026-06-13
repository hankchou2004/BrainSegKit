# scripts/ — 技能 / Skills

## viewer.py — 互動式影像檢視器

```bash
cd /home/hank/medical_segmention/MedSegKit
conda activate brain_segmention

# 互動模式（需本機 display 或 WSLg）
python scripts/viewer.py --dataset oasis1 --split train
python scripts/viewer.py --dataset btcv   --split val

# 無頭 PNG 模式（SSH 可用）
python scripts/viewer.py --dataset oasis1 --save --case 0 --save-dir ./viewer_out
python scripts/viewer.py --dataset btcv   --save --case 3 --save-dir ./viewer_out
```

### 快捷鍵 / Shortcuts

| 按鍵 | 功能 |
|------|------|
| `← / →` | 切換 case |
| `A / N` | 全開 / 全關所有 labels |
| `Click`（panel） | 切換單一 label 顯示 |
| `Scroll` | 縮放 |

**OASIS-1**：`H`=Hippocampus、`V`=Ventricles、`C`=Cortex、`1/2/3`=WM/GM/CSF  
**BTCV**：`L`=Liver、`S`=Spleen、`K`=Kidneys、`P`=Pancreas、`A`=Aorta、`1/2/3`=組合

## convert_dataset.py — OASIS-1 資料轉換

```bash
conda activate brain_segmention
python scripts/convert_dataset.py
# brain_data/ → dataset/oasis1_freesurfer/（train 255 / val 85 / test 85）
```

- 只需執行一次
- 需要 `brain_data/` 存在，執行後自動刪除
- 輸出：`imagesTr/`、`labelsTr/`、`imagesTs/`、`labelsTs/`、`splits.json`
