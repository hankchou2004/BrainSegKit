# medsegkit/engine/ — Claude Code 上下文

## 用途 / Purpose

Lightning `LightningModule` 封裝訓練邏輯。`SegModule` 處理一般分割訓練；`KDModule` 處理知識蒸餾。

Lightning `LightningModule` wrapping training logic. `SegModule` for standard segmentation; `KDModule` for knowledge distillation.

## 檔案說明 / Files

| 檔案 | 用途 |
|------|------|
| `seg_module.py` | `SegModule`：一般分割，含 sliding-window inference、Dice+HD95 |
| `kd_module.py` | `KDModule`：知識蒸餾，teacher 凍結，student 訓練 |

## SegModule 重要細節 / SegModule Key Details

- `forward()` 直接呼叫 `self.model(x)`
- `_shared_eval()` 用 `SlidingWindowInferer`（overlap=0.5, mode="gaussian"）
- DynUNet deep supervision：`training_step` 自動偵測回傳 list，取平均損失
- `save_hyperparameters()` 保存所有 `__init__` 參數到 checkpoint
- `num_classes` 預設從 `transforms.py` 的 `NUM_CLASSES`（41）；BTCV 需在 config 的 `out_channels` 指定 14

## KDModule 重要細節 / KDModule Key Details

- Teacher 模型從 checkpoint 載入後 `requires_grad_(False)`（完全凍結）
- Student 模型正常訓練
- `kd_type` 決定使用哪個損失函數（`losses/kd/` 目錄）

## checkpoint 載入 / Loading Checkpoints

```python
from medsegkit.engine.seg_module import SegModule
module = SegModule.load_from_checkpoint("logs/.../best_model.ckpt")
model = module.model  # 取出 nn.Module
```
