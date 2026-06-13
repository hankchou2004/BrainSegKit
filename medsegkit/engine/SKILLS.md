# medsegkit/engine/ — 技能 / Skills

## SegModule

```python
from medsegkit.engine.seg_module import SegModule

# 建立（訓練用）
module = SegModule(
    model_name="dynunet",
    model_kwargs={"in_channels": 1, "out_channels": 14},
    loss_name="dice_ce",
    lr=1e-4,
    weight_decay=1e-5,
    patch_size=(96, 96, 96),
    sw_batch_size=4,
    num_classes=14,
)

# 從 checkpoint 載入（推論/評估用）
module = SegModule.load_from_checkpoint(
    "logs/dynunet_btcv/best_model.ckpt",
    map_location="cuda",
)
model = module.model  # torch.nn.Module

# 手動推論
import torch
from monai.inferers import SlidingWindowInferer
inferer = SlidingWindowInferer(roi_size=(96,96,96), sw_batch_size=4, overlap=0.5)
with torch.no_grad():
    logits = inferer(inputs=image, network=model)  # (1, C, H, W, D)
pred = logits.argmax(dim=1)  # (1, H, W, D)
```

## KDModule

```python
from medsegkit.engine.kd_module import KDModule

module = KDModule(
    teacher_name="dynunet",
    teacher_ckpt="/path/to/teacher.ckpt",
    student_name="unet",
    student_kwargs={"in_channels": 1, "out_channels": 14},
    kd_type="response",      # response | feature | contrastive | combined
    kd_weight=0.5,
    temperature=4.0,
    seg_loss_name="dice_ce",
    lr=1e-4,
)
```

## 指標說明 / Metrics

訓練過程中自動記錄：

| 指標 | 說明 |
|------|------|
| `train/loss` | 訓練損失（每 step + 每 epoch） |
| `val/dice` | 驗證 Dice（`include_background=False`，多類別平均） |
| `val/hd95` | 驗證 Hausdorff Distance 95% |
| `test/dice` | 測試 Dice |
| `test/hd95` | 測試 HD95 |
