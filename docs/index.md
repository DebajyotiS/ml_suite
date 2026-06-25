# ML Suite

Reusable, dimension-agnostic deep learning building blocks for generative modelling research. Build classifiers, diffusion models, flow-matching networks, and point-cloud models from composable PyTorch primitives that work identically across 1D, 2D, and 3D data.

---

## Where to start

Pick the tab that matches your goal. Each one shows the recommended model and a minimal working example. The API links at the bottom of each tab have the full parameter reference.

=== "I want to classify something"

    Use **`ConvNet`** when your input is a spatial grid like images, 1D signals, or 3D volumes. Use **`make_point_cloud_classifier`** when your input is an unordered set of points, since the transformer is permutation-invariant and order does not matter.

    Both return a logit vector of shape `(batch, num_classes)`.

    ```python
    from ml_suite.models.convolution import ConvNet
    from ml_suite.models.transformer.presets import make_point_cloud_classifier

    # 2-D image classifier (e.g. CIFAR-10)
    model = ConvNet(conv_dim=2, in_channels=3, stage_channels=[64, 128, 256],
                    blocks_per_stage=2, num_classes=10)

    # Permutation-invariant point-cloud classifier (e.g. ModelNet40)
    model = make_point_cloud_classifier(point_dim=3, num_classes=40,
                                        embedding_dim=256, depth=6, num_heads=4)
    ```

    See: [Convolution](api/convolution.md) · [Transformer presets](api/transformer/presets.md)

=== "I want a diffusion / flow model"

    These models learn to predict a denoising direction (DDPM) or a velocity field (flow matching) and require a **time embedding** at every forward pass.

    Use **`ConditionedUNet`** for spatial data with local structure like images or voxels. Use **`make_conditioned_point_to_point_model`** for point sets or sequences where you want a per-token output of the same shape as the input.

    ```python
    from ml_suite.models.unet import ConditionedUNet
    from ml_suite.models.transformer.presets import make_conditioned_point_to_point_model

    # Time-conditioned 2-D U-Net (DDPM / EDM style). Output shape matches input.
    model = ConditionedUNet(conv_dim=2, in_channels=3, out_channels=3,
                            stage_channels=[64, 128, 256, 512],
                            time_conditioning=True)
    out = model(x, time=t)  # x: (B, 3, H, W), t: (B,)

    # Flow-matching velocity field on point sets. Output shape matches input.
    model = make_conditioned_point_to_point_model(
        point_dim=3, output_dim=3, embedding_dim=128,
        depth=6, num_heads=4, time_conditioning=True)
    out = model(x, time=t)  # x: (B, N, 3), t: (B,)
    ```

    See: [U-Net](api/unet.md) · [Transformer presets](api/transformer/presets.md)

=== "I want a patch / ViT model"

    **`PatchTransformerND`** splits a spatial grid into non-overlapping patches, encodes them with a transformer, and can produce different output shapes depending on your task:

    - **Grid reconstruction** (`make_patch_grid_model`): output has the same spatial shape as the input, useful for segmentation or denoising backbones.
    - **Classification** (`make_patch_classifier`): a single pooled class vector, the standard ViT setup.

    Both work on 1-D, 2-D, and 3-D grids without changing the API.

    ```python
    from ml_suite.models.transformer.presets import make_patch_grid_model, make_patch_classifier

    # ViT-style image-to-image (e.g. segmentation backbone)
    model = make_patch_grid_model(input_dim=2, in_channels=3, out_channels=1,
                                  patch_size=16, embedding_dim=256,
                                  depth=12, num_heads=8)

    # ViT classifier (e.g. ImageNet)
    model = make_patch_classifier(input_dim=2, in_channels=3, num_classes=1000,
                                  patch_size=16, embedding_dim=768,
                                  depth=12, num_heads=12)
    ```

    See: [Transformer presets](api/transformer/presets.md) · [Transformer models](api/transformer/models.md)

=== "I want to build something custom"

    The preset functions are thin wrappers around composable primitives. If you need a non-standard architecture with mixed modalities, custom conditioning, or unusual head shapes, you can wire the pieces together directly.

    The typical stack is **tokenizer -> transformer -> head**, with optional conditioning injected at the transformer level via FiLM or cross-attention.

    ```python
    from ml_suite.models.transformer import (
        ContinuousInputTokenizer, TransformerStack,
        PooledHead, RotaryEmbedding,
    )
    from ml_suite.models.linear import MLP

    tokenizer = ContinuousInputTokenizer(input_dim=64, embedding_dim=256)
    stack = TransformerStack(embedding_dim=256, depth=6, num_heads=8)
    head = PooledHead(embedding_dim=256, output_dim=10)

    tokens = tokenizer(x)       # (B, N, 256)
    encoded = stack(tokens)     # (B, N, 256)
    out = head(encoded)         # (B, 10)
    ```

    See: [Transformer primitives](api/transformer/primitives.md) · [Linear](api/linear.md)

---

## Module map

| Module | What it provides | Dimensionality |
|---|---|---|
| [`models.linear`](api/linear.md) | Linear blocks, MLPs, FiLM / cross-attn conditioning | any |
| [`models.convolution`](api/convolution.md) | Conv blocks, separable variants, multi-stage backbones | 1D / 2D / 3D |
| [`models.unet`](api/unet.md) | Encoder-decoder U-Nets with skip connections and rich conditioning | 1D / 2D / 3D |
| [`models.transformer`](api/transformer/models.md) | Token-centric transformers: classification, generation, patch grids | any |
| [`models.transformer.presets`](api/transformer/presets.md) | One-call factory functions for common configurations | any |
| [`utils`](api/utils.md) | Activation factory, sinusoidal / MLP time embeddings | any |

---

## Installation

```bash
uv sync --extra dev   # recommended
pip install -e ".[dev]"  # pip alternative
```

Requires Python >= 3.13 and PyTorch >= 2.12.
