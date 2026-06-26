<div align="center">

<img src="assets/ml_suite_rainbow.png" alt="ml_suite" width="480">

<br/>
<br/>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.12%2B-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml/badge.svg"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/build.yml"><img alt="Build" src="https://github.com/DebajyotiS/ml_suite/actions/workflows/build.yml/badge.svg"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml"><img alt="Coverage" src="https://raw.githubusercontent.com/DebajyotiS/ml_suite/coverage-data/badge.svg"></a>
<a href="https://DebajyotiS.github.io/ml_suite/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white"></a>

</div>

---

ml_suite is a set of reusable PyTorch building blocks for generative modelling research. The same classes handle 1D signals, 2D images, and 3D volumes. You pick a dimensionality at construction time and the rest of the code stays the same.

---

## Quick example

```python
import torch
import torch.nn.functional as F
from ml_suite.models.unet import ConditionedUNet

# Flow-matching velocity estimator for 2D images
model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256, 512],
    time_conditioning=True,
    attention_downsample_factors=[4, 8],
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 4
x = torch.randn(batch_size, 3, 64, 64)   # real data
noise = torch.randn_like(x)
t = torch.rand(batch_size)
t_view = t.view(-1, 1, 1, 1)

x_t = (1 - t_view) * noise + t_view * x
velocity_pred = model(x_t, time=t)
loss = F.mse_loss(velocity_pred, x - noise)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Switch to 1D by changing `conv_dim=1` and passing a `(batch, channels, length)` tensor. No other code changes.

---

## Module map

| Module | What it provides | Dimensionality | Typical use |
|---|---|---|---|
| `models.linear` | Linear blocks, MLPs, FiLM / cross-attn conditioning | any | Conditioning blocks and MLP heads |
| `models.convolution` | Conv blocks, separable variants, multi-stage backbones | 1D / 2D / 3D | Classifiers on spatial data |
| `models.unet` | Encoder-decoder U-Nets with skip connections and rich conditioning | 1D / 2D / 3D | Generation and reconstruction |
| `models.transformer` | Token-centric transformers: classification, generation, patch grids | any | Sequences, point clouds, and patches |
| `models.transformer.presets` | One-call factory functions for common configurations | any | Quick model construction |
| `utils` | Activation factory, sinusoidal / MLP time embeddings | any | Time embeddings for diffusion models |

---

## Installation

```bash
git clone https://github.com/DebajyotiS/ml_suite
cd ml_suite
uv sync --extra dev   # recommended
# pip install -e ".[dev]"  # pip alternative
```

Requires Python >= 3.13 and PyTorch >= 2.12.

---

## Documentation

Full API reference, getting started guide, and recipes: **[DebajyotiS.github.io/ml_suite](https://DebajyotiS.github.io/ml_suite/)**

---

## Development

```bash
pytest
ruff check src
ruff format --check src
```
