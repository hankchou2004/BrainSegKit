# medsegkit/models/ — 技能 / Skills

## Registry API

```python
from medsegkit.models.registry import build_model, list_models, register_model

# 查詢所有模型
list_models()
# ['attention_unet', 'dynunet', 'medsam', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# 建立模型
model = build_model("unet",     in_channels=1, out_channels=41)  # OASIS-1
model = build_model("dynunet",  in_channels=1, out_channels=14)  # BTCV
model = build_model("medsam",   in_channels=1, out_channels=14,
                    checkpoint="/path/to/medsam_vit_b.pth",
                    freeze_encoder=True, slice_axis=2, slice_batch=8)
model = build_model("umamba",   in_channels=1, out_channels=14)  # 需 mamba-ssm
```

## 模型參數量 / Model Parameters

| 模型 | 參數量 | 備註 |
|------|--------|------|
| `unet` | ~19 M | 全部可訓練 |
| `unet_pp` | ~9 M | 全部可訓練 |
| `attention_unet` | ~12 M | 全部可訓練 |
| `dynunet` | ~20 M | 全部可訓練，deep supervision |
| `swin_unetr` | ~62 M | 全部可訓練 |
| `mednext` | ~18 M | 全部可訓練 |
| `segresnet` | ~5 M | 最輕量 |
| `unetr` | ~93 M | 最大 MONAI 模型 |
| `medsam` | 90 M（0.16 M trainable） | ViT-B 凍結，只訓練 decoder |
| `umamba` | ~20 M | 全部可訓練，需額外安裝 |

## 自訂模型範例 / Custom Model Example

```python
from medsegkit.models.registry import register_model
import torch.nn as nn

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, 1),
        )
    def forward(self, x):
        return self.net(x)
```
