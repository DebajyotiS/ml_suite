<div align="center">

<img src="assets/ml_suite_rainbow.png" alt="ml_suite" width="480">

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Tests](https://github.com/DebajyotiS/generative_models/actions/workflows/tests.yml/badge.svg)](https://github.com/DebajyotiS/generative_models/actions/workflows/tests.yml)
[![Build](https://github.com/DebajyotiS/generative_models/actions/workflows/build.yml/badge.svg)](https://github.com/DebajyotiS/generative_models/actions/workflows/build.yml)
[![Coverage](https://raw.githubusercontent.com/DebajyotiS/generative_models/coverage-data/badge.svg)](https://github.com/DebajyotiS/generative_models/actions/workflows/tests.yml)

A PyTorch library of reusable, dimension-agnostic deep learning building blocks for generative modelling research. It supplies composable primitives such as linear blocks, convolution stacks, U-Nets, and token-centric transformers with a consistent API across 1D, 2D, and 3D data.

</div>

---

## Requirements

- Python >= 3.13
- PyTorch >= 2.12

---

## Installation

Using pip in editable mode:

```bash
pip install -e ".[dev]"
```

Using uv:

```bash
uv sync extra --dev
```

---

## Quick start

### Linear / MLP

```python
from ml_suite.models.linear import MLP

# 3-layer MLP with SiLU activations
model = MLP(input_dim=64, hidden_dim=128, num_layers=3)
out = model(x)  # (..., 64) -> (..., 128)

# With FiLM conditioning
model = MLP(input_dim=64, hidden_dim=128, num_layers=3, context_dim=32, context_injection="film")
out = model(x, context=c)
```

### Convolution

```python
from ml_suite.models.convolution import ConvNet, ConditionedConvNet

# 2D feature extractor
backbone = ConvNet(conv_dim=2, in_channels=3, stage_channels=[64, 128, 256], blocks_per_stage=2)

# 2D classifier
classifier = ConvNet(
    conv_dim=2, in_channels=3, stage_channels=[64, 128, 256], blocks_per_stage=2, num_classes=10
)

# FiLM-conditioned backbone
conditioned = ConditionedConvNet(
    conv_dim=2, in_channels=3, stage_channels=[64, 128, 256], blocks_per_stage=2, context_dim=128
)
out = conditioned(x, context=c)
```

### U-Net

```python
from ml_suite.models.unet import UNet, ConditionedUNet

# Plain 2D U-Net
model = UNet(conv_dim=2, in_channels=3, out_channels=1, stage_channels=[64, 128, 256, 512])
out = model(x)  # (B, 3, H, W) -> (B, 1, H, W)

# Time-conditioned U-Net for diffusion
model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256, 512],
    time_conditioning=True,
)
out = model(x, time=t)  # t has shape (B,)

# With class conditioning and cross-attention
model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256, 512],
    time_conditioning=True,
    num_classes=10,
    attention_downsample_factors=[4, 8],
    attention_type="self_cross",
    cross_attention_dim=512,
)
out = model(x, time=t, class_labels=labels, cross_context=tokens)
```

### Transformer

```python
from ml_suite.models.transformer import (
    TokenToTokenTransformer,
    TokenToClassTransformer,
    ConditionedTokenTransformer,
    PatchTransformerND,
)

# Per-token output (sequence prediction, set-to-set)
model = TokenToTokenTransformer(
    input_dim=64, output_dim=64, embedding_dim=128, depth=6, num_heads=4
)
out = model(x)  # (B, T, 64) -> (B, T, 64)

# Classification
model = TokenToClassTransformer(
    input_dim=64, num_classes=10, embedding_dim=128, depth=6, num_heads=4, max_length=512
)
logits = model(x)  # (B, T, 64) -> (B, 10)

# Conditional (diffusion / score matching)
model = ConditionedTokenTransformer(
    input_dim=3, output_dim=3, embedding_dim=128, depth=6, num_heads=4, time_conditioning=True
)
out = model(x, time=t)

# ViT-style patch transformer
model = PatchTransformerND(
    input_dim=2, in_channels=3, out_channels=1, patch_size=16,
    embedding_dim=256, depth=12, num_heads=8, output_mode="grid"
)
out = model(x)  # (B, 3, H, W) -> (B, 1, H, W)
```

### Preset factory functions

```python
from ml_suite.models.transformer.presets import (
    make_point_cloud_classifier,
    make_point_to_point_model,
    make_conditioned_point_to_point_model,
    make_sequence_classifier,
    make_patch_grid_model,
    make_patch_classifier,
)

# Permutation-invariant point-cloud classifier
clf = make_point_cloud_classifier(point_dim=3, num_classes=40, embedding_dim=256, depth=6, num_heads=4)

# Diffusion / flow-matching velocity field on point sets
velocity = make_conditioned_point_to_point_model(
    point_dim=3, output_dim=3, embedding_dim=128, depth=6, num_heads=4, time_conditioning=True
)
```

---

## Module overview

### Linear models  `ml_suite.models.linear`

| Class | Description |
|---|---|
| `LinearBlock` | Single linear projection with optional context conditioning (concat, add, multiply, FiLM, cross-attn) and residual connection. |
| `MLP` | Sequential stack of `LinearBlock` layers. |
| `VADLinearBlock` | Variational adaptive dropout linear block (placeholder, not yet implemented). |

### Convolution models  `ml_suite.models.convolution`

| Class | Description |
|---|---|
| `ConvBlock` | Standard Conv + Norm + Activation block, dimension-agnostic (1D/2D/3D). |
| `ConditionedConvBlock` | FiLM-conditioned `ConvBlock`. |
| `SeparableConvBlock` | Depthwise-separable conv block. |
| `SeparableConditionedConvBlock` | FiLM-conditioned separable block. |
| `ConvNet` | Multi-stage convolutional backbone or classifier. |
| `ConditionedConvNet` | FiLM-conditioned `ConvNet`. |

### U-Net models `ml_suite.models.unet`

| Class | Description |
|---|---|
| `UNet` | Plain encoder-decoder U-Net for 1D, 2D, or 3D data with optional self-attention. |
| `ConditionedUNet` | FiLM-conditioned U-Net with support for time, class, global context, and cross-attention conditioning. |

### Transformer models `ml_suite.models.transformer`

| Class | Description |
|---|---|
| `TokenToTokenTransformer` | Per-token output; sequence-to-sequence and set-to-set tasks. |
| `TokenToVectorTransformer` | Pools tokens to a single vector; regression tasks. |
| `TokenToClassTransformer` | Specialisation of `TokenToVectorTransformer` for classification. |
| `ConditionedTokenTransformer` | Conditional token-to-token model with time, class, global context, and cross-attention. |
| `PatchTransformerND` | ViT-style patch transformer for 1D, 2D, or 3D grids; supports grid, token, and vector outputs. |

### Utilities `ml_suite.utils`

| Module | Description |
|---|---|
| `activations.get_activation` | Activation factory: relu, leakyrelu, gelu, sigmoid, tanh, softmax, silu/swish, mish, identity. |
| `conditioning.SinusoidalTimeEmbedding` | Fixed sinusoidal time embedding (Transformer-style, suitable for diffusion). |
| `conditioning.TimeEmbeddingMLP` | Projects scalar timesteps to a target embedding dimension. |
| `samplers.ode` | ODE-based sampling utilities (planned). |
| `samplers.sde` | SDE-based sampling utilities (planned). |

---

## Notebooks

Four reference notebooks are in `notebooks/`:

| Notebook | Topic |
|---|---|
| `01_linear_models.ipynb` | LinearBlock conditioning modes, MLP construction, broadcasting behaviour |
| `02_convolution_models.ipynb` | ConvNet stages, downsampling modes, receptive field diagnostics, 1D/2D/3D examples |
| `03_unet_models.ipynb` | UNet architecture, skip connections, cross-attention, time embedding integration |
| `04_transformer_models.ipynb` | Tokenizers, positional encodings, pooling, heads, conditioning, preset models |

---

## Development

```bash
# Run tests
pytest

# Lint and format checks
ruff check src
ruff format --check src
```
