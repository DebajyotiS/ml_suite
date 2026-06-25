# Getting Started

## Installation

=== "uv (recommended)"

    ```bash
    git clone https://github.com/DebajyotiS/ml_suite
    cd ml_suite
    uv sync --extra dev
    ```

=== "pip"

    ```bash
    git clone https://github.com/DebajyotiS/ml_suite
    cd ml_suite
    pip install -e ".[dev]"
    ```

**Requirements:** Python >= 3.13, PyTorch >= 2.12

---

## Core concepts

ml_suite is organised into four model families. Every family is dimension-agnostic — you pick `1`, `2`, or `3` at construction time and the same class handles signals, images, or volumes.

| Family | Entry point | Key parameter |
|---|---|---|
| Linear / MLP | `MLP`, `LinearBlock` | `input_dim`, `output_dim` |
| Convolution | `ConvNet`, `ConditionedConvNet` | `conv_dim` ∈ {1, 2, 3} |
| U-Net | `UNet`, `ConditionedUNet` | `conv_dim` ∈ {1, 2, 3} |
| Transformer | `TokenToToken…`, `PatchTransformerND` | `input_dim` ∈ {1, 2, 3} (patch only) |

Conditioning follows the same pattern everywhere: pass `time`, `class_labels`, `global_context`, or `cross_context` to the `forward` call of any `Conditioned*` model.

---

## Example: image classifier

```python
import torch
from ml_suite.models.convolution import ConvNet

model = ConvNet(
    conv_dim=2,
    in_channels=3,
    stage_channels=[64, 128, 256, 512],
    blocks_per_stage=2,
    num_classes=10,
)

x = torch.randn(8, 3, 64, 64)
logits = model(x)   # (8, 10)
```

---

## Example: diffusion U-Net

```python
import torch
from ml_suite.models.unet import ConditionedUNet

model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256, 512],
    time_conditioning=True,
    attention_downsample_factors=[4, 8],
)

x = torch.randn(4, 3, 64, 64)
t = torch.rand(4)
out = model(x, time=t)   # (4, 3, 64, 64)
```

---

## Example: point-cloud flow model

```python
import torch
from ml_suite.models.transformer.presets import make_conditioned_point_to_point_model

model = make_conditioned_point_to_point_model(
    point_dim=3,
    output_dim=3,
    embedding_dim=128,
    depth=6,
    num_heads=4,
    time_conditioning=True,
)

x = torch.randn(4, 1024, 3)   # (batch, points, xyz)
t = torch.rand(4)
velocity = model(x, time=t)   # (4, 1024, 3)
```

---

## Example: ViT patch model

```python
import torch
from ml_suite.models.transformer import PatchTransformerND

# Patch-based image-to-image model (e.g. denoiser or segmentation head)
model = PatchTransformerND(
    input_dim=2,
    in_channels=3,
    out_channels=1,
    patch_size=16,
    embedding_dim=256,
    depth=6,
    num_heads=8,
    output_mode="grid",
)

x = torch.randn(2, 3, 256, 256)
out = model(x)   # (2, 1, 256, 256)
```

---

## Conditioning cheat sheet

All `Conditioned*` models accept the same optional kwargs in `forward`:

| Argument | Shape | When to use |
|---|---|---|
| `time` | `(batch,)` | Diffusion / flow timestep |
| `class_labels` | `(batch,)` int | Class-conditional generation |
| `global_context` | `(batch, context_dim)` | Any global vector embedding |
| `cross_context` | `(batch, tokens, dim)` | Text / cross-attention conditioning |
| `cross_context_mask` | `(batch, tokens)` bool | Padding mask for `cross_context` |

---

## Next steps

- [API Reference](api/linear.md) — full class and method documentation
- [Transformer primitives](api/transformer/primitives.md) — building blocks for custom architectures
- [Preset factory functions](api/transformer/presets.md) — one-liner model construction
