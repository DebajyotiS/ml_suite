<div align="center">

<img src="../assets/ml_suite_rainbow.png" alt="ml_suite" width="480">

</div>

A PyTorch library of reusable, dimension-agnostic deep learning building blocks for generative modelling research. It supplies composable primitives such as linear blocks, convolution stacks, U-Nets, and token-centric transformers with a consistent API across 1D, 2D, and 3D data.

---

## Requirements

- Python >= 3.13
- PyTorch >= 2.12

---

## Installation

=== "uv"

    ```bash
    uv sync --extra dev
    ```

=== "pip"

    ```bash
    pip install -e ".[dev]"
    ```

---

## Module overview

| Namespace | What it provides |
|---|---|
| [`ml_suite.models.linear`](api/linear.md) | Linear blocks and MLPs with optional FiLM / cross-attention conditioning |
| [`ml_suite.models.convolution`](api/convolution.md) | Dimension-agnostic conv blocks, separable variants, and multi-stage backbones |
| [`ml_suite.models.unet`](api/unet.md) | Encoder-decoder U-Nets with skip connections, self-attention, and rich conditioning |
| [`ml_suite.models.transformer`](api/transformer/models.md) | Token-centric transformers: token-to-token, classification, diffusion, and patch grids |
| [`ml_suite.models.transformer.presets`](api/transformer/presets.md) | One-call factory functions for common transformer configurations |
| [`ml_suite.utils`](api/utils.md) | Activation factory, sinusoidal / learned time embeddings |
