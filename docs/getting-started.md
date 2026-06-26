# Getting Started

By the end of this page you will be able to construct, run a forward pass through, and take a training step with any of the four model families in ml_suite. The examples assume basic familiarity with PyTorch tensors and `nn.Module`. You only need to read the sections that match your task.

---

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

ml_suite is organised into four model families. Every family is dimension-agnostic. You pick `1`, `2`, or `3` at construction time and the same class handles signals, images, or volumes.

| Family | Entry point | Key parameter |
|---|---|---|
| Linear / MLP | `MLP`, `LinearBlock` | `input_dim`, `output_dim` |
| Convolution | `ConvNet`, `ConditionedConvNet` | `conv_dim` in {1, 2, 3} |
| U-Net | `UNet`, `ConditionedUNet` | `conv_dim` in {1, 2, 3} |
| Transformer | `TokenToToken...`, `PatchTransformerND` | `input_dim` in {1, 2, 3} (patch only) |

Conditioning follows the same pattern everywhere: pass `time`, `class_labels`, `global_context`, or `cross_context` to the `forward` call of any `Conditioned*` model.

---

## Example: image classifier

```python
import torch
import torch.nn.functional as F
from ml_suite.models.convolution import ConvNet

model = ConvNet(
    conv_dim=2,
    in_channels=3,
    stage_channels=[64, 128, 256, 512],
    blocks_per_stage=2,
    num_classes=10,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(8, 3, 64, 64)
labels = torch.randint(0, 10, (8,))
logits = model(x)   # (8, 10)

loss = F.cross_entropy(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())  # should be around 2.3 for random data, just verifying the forward and backward pass run
```

---

## Example: diffusion U-Net

```python
import torch
import torch.nn.functional as F
from ml_suite.models.unet import ConditionedUNet

model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256, 512],
    time_conditioning=True,
    attention_downsample_factors=[4, 8],
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(4, 3, 64, 64)
t = torch.rand(4)
target = torch.randn(4, 3, 64, 64)
out = model(x, time=t)   # (4, 3, 64, 64)

loss = F.mse_loss(out, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())  # should be around 2.0 for random data, just verifying the forward and backward pass run
```

---

## Example: point-cloud flow model

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer.presets import make_conditioned_point_to_point_model

model = make_conditioned_point_to_point_model(
    point_dim=3,
    output_dim=3,
    embedding_dim=128,
    depth=6,
    num_heads=4,
    time_conditioning=True,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(4, 1024, 3)   # (batch, points, xyz)
t = torch.rand(4)
target = torch.randn(4, 1024, 3)
velocity = model(x, time=t)   # (4, 1024, 3)

loss = F.mse_loss(velocity, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())  # should be around 2.0 for random data, just verifying the forward and backward pass run
```

---

## Example: ViT patch model

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer import PatchTransformerND

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(2, 3, 256, 256)
target = torch.randn(2, 1, 256, 256)
out = model(x)   # (2, 1, 256, 256)

loss = F.mse_loss(out, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())  # should be around 2.0 for random data, just verifying the forward and backward pass run
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

## Known gotchas

1. **Group norm channel counts.** Group norm requires channel counts divisible by `num_groups`. The default is `num_groups=32`. If your `stage_channels` values are not divisible by 32 you will get a wrong-shape error. Either set `num_groups=1` or pick channels that are multiples of 32 (64, 128, 256 work).

2. **Attention is off by default.** `attention_downsample_factors` defaults to an empty tuple, which means no attention is added. This is correct for quick tests, but real diffusion models typically add attention at the bottleneck. Set it to something like `[4, 8]` to add attention at those downsample factors.

3. **Time is always required when enabled.** For `ConditionedUNet` with `time_conditioning=True`, you must pass `time=t` at every forward call. Omitting it does not raise an error at construction time.

4. **Cross-attention context shape.** `context_injection='cross_attn'` in `LinearBlock` expects a context tensor with a sequence dimension: shape `(batch, tokens, dim)`. All other injection modes expect `(batch, dim)`. Passing the wrong shape will silently broadcast incorrectly.

---

## Where to go from here

If you want to read the full parameter reference for a specific component, go to the [API reference](api/linear.md). If you want to wire primitives together without using a preset, start with the [transformer primitives page](api/transformer/primitives.md). The [preset factory functions page](api/transformer/presets.md) shows all one-liner constructors.
