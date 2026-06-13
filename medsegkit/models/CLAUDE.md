# medsegkit/models/ — Claude Code 上下文

## 用途 / Purpose

模型 Registry 及所有模型實作。`registry.py` 提供 `@register_model` 裝飾器；`wrappers.py` 包裝 8 個 MONAI 模型；自製模型（`medsam.py`、`umamba/`）各自實作 `nn.Module`。

Model Registry and all model implementations. `registry.py` provides `@register_model`; `wrappers.py` wraps 8 MONAI models; custom models (`medsam.py`, `umamba/`) implement `nn.Module` directly.

## 檔案說明 / Files

| 檔案 | 說明 |
|------|------|
| `registry.py` | `build_model()`, `list_models()`, `register_model()` |
| `wrappers.py` | UNet / UNet++ / AttentionUNet / DynUNet / SwinUNETR / MedNeXt / SegResNet / UNETR |
| `medsam.py` | MedSAM：SAM ViT-B encoder（凍結） + 輕量 2-layer ConvTranspose2d decoder |
| `umamba/umamba.py` | U-Mamba：Mamba SSM bottleneck |
| `__init__.py` | import 所有模型以觸發 `@register_model` |

## 新增模型 / Adding a New Model

1. 建立 `medsegkit/models/my_model.py`
2. 實作 `nn.Module`，`__init__` 必須接受 `in_channels` 和 `out_channels`
3. 加上 `@register_model("my_key")` 裝飾器
4. 在 `medsegkit/models/__init__.py` 加入 `from medsegkit.models import my_model  # noqa: F401`

## 模型介面規範 / Model Interface

```python
@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, **kw):
        # **kw 吸收 config 中多餘的 YAML 欄位
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        # 回傳: (B, out_channels, H, W, D)
        # DynUNet deep supervision: 可回傳 list[Tensor]
        ...
```

## MedSAM 特殊說明 / MedSAM Notes

- `_SAM_INPUT_SIZE = 1024`（ViT-B 固定輸入大小）
- `_SAM_EMBED_DIM  = 256`（neck 輸出 channels）
- `_SAM_FEAT_SIZE  =  64`（neck 輸出空間大小）
- Decoder：`256×64×64 → 128×128 → 64×256×256 → C×H×W`，最後 resize 回原始 slice 大小
- `freeze_encoder=True` 時 encoder 完全不計算梯度（使用 `torch.no_grad()`）
- OOM 時降低 `slice_batch`（預設 16，建議從 8 開始）
