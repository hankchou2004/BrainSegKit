# medsegkit/models/umamba/ — Claude Code 上下文

## 用途 / Purpose

U-Mamba 自製實作。使用 Mamba SSM（State Space Model）作為 bottleneck，替代傳統的 Transformer attention。

Custom U-Mamba implementation using Mamba SSM as the bottleneck, replacing traditional Transformer attention.

## 依賴 / Dependencies

```bash
pip install mamba-ssm causal-conv1d
# 或
pip install "medsegkit[umamba]"
```

需要 CUDA、不支援 CPU 推論。

## 重要注意 / Important Notes

- `mamba-ssm` 安裝需要 CUDA toolkit，在純 CPU 環境無法安裝
- `@register_model("umamba")` 已在 `__init__.py` 透過 lazy import 保護：沒有 `mamba-ssm` 時跳過，不影響其他模型
- 如果 import 失敗，`list_models()` 不會包含 `"umamba"`
