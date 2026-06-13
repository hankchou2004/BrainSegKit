# scripts/ — Claude Code 上下文

## 用途 / Purpose

資料預處理與互動視覺化的獨立腳本。不依賴 Lightning Trainer，直接操作檔案系統或 matplotlib。

Standalone scripts for data preprocessing and interactive visualization. Independent of Lightning Trainer.

## 腳本說明 / Scripts

| 腳本 | 用途 |
|------|------|
| `convert_dataset.py` | 將 OASIS-1 FreeSurfer `.mgz` 轉換為 nnUNet `.nii.gz` 格式 |
| `viewer.py` | 互動式多軸向影像 + 分割標籤檢視器 |

## viewer.py 重要細節 / viewer.py Key Details

- `--dataset oasis1 | btcv`（必填）
- `--split train | val | test`（預設 `train`）
- `--save`：SSH 無頭模式，不開視窗，輸出 PNG
- WSL2 需設定 `DISPLAY=:0`，或用 `--save` 繞過
- Panel 點擊使用 `button_press_event` + `_row_y_map` hit-test（整行都可點擊）
- 不要改回 `pick_event`（太小難點到）

## convert_dataset.py 重要細節 / convert_dataset.py Key Details

- 輸入：`/home/hank/medical_segmention/brain_data/` 的原始 `.mgz` 檔
- 輸出：`/home/hank/medical_segmention/dataset/oasis1_freesurfer/`（nnUNet 格式）
- FreeSurfer aseg 標籤重映射：非連續值 → 0–40（詳見 `medsegkit/data/transforms.py`）
- 執行完成後 `brain_data/` 會被刪除
