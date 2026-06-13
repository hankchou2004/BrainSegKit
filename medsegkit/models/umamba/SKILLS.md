# medsegkit/models/umamba/ — 技能 / Skills

## 使用 U-Mamba / Using U-Mamba

```bash
# 安裝依賴
pip install "medsegkit[umamba]"

# 確認已註冊
python -c "import medsegkit; print(medsegkit.list_models())"
# 應包含 'umamba'
```

```python
from medsegkit.models.registry import build_model

model = build_model("umamba", in_channels=1, out_channels=14)
```

## 訓練設定 / Training Config

```yaml
model:
  name: umamba
  in_channels: 1
  out_channels: 14     # BTCV
  # out_channels: 41   # OASIS-1

data:
  dataset: btcv
  patch_size: [96, 96, 96]
  batch_size: 2

training:
  max_epochs: 500
  lr: 1.0e-4
```

## U-Mamba vs Transformer / Comparison

| | SwinUNETR | U-Mamba |
|--|-----------|---------|
| 序列建模 | Window Attention（O(n²)） | SSM（O(n)） |
| 長距離依賴 | 受限於 window size | 線性複雜度，可處理長序列 |
| 參數量 | ~62 M | ~20 M |
| 需要額外安裝 | 否 | 是（mamba-ssm） |
