# ML Suite

Reusable, dimension-agnostic deep learning building blocks for generative modelling research. Build classifiers, diffusion models, flow-matching networks, and point-cloud models from composable PyTorch primitives that work identically across 1D, 2D, and 3D data.

---

## Where to start

=== "I want to classify something"

    **ConvNet** for images or signals, **TokenToClassTransformer** for sets and sequences.

    ```python
    from ml_suite.models.convolution import ConvNet
    from ml_suite.models.transformer.presets import make_point_cloud_classifier

    # 2-D image classifier
    model = ConvNet(conv_dim=2, in_channels=3, stage_channels=[64, 128, 256],
                    blocks_per_stage=2, num_classes=10)

    # Permutation-invariant point-cloud classifier
    model = make_point_cloud_classifier(point_dim=3, num_classes=40,
                                        embedding_dim=256, depth=6, num_heads=4)
    ```

    See: [Convolution](api/convolution.md) · [Transformer presets](api/transformer/presets.md)

=== "I want a diffusion / flow model"

    **ConditionedUNet** for spatial data, **ConditionedTokenTransformer** or the conditioned preset for sets and point clouds.

    ```python
    from ml_suite.models.unet import ConditionedUNet
    from ml_suite.models.transformer.presets import make_conditioned_point_to_point_model

    # Time-conditioned 2-D U-Net (DDPM / EDM style)
    model = ConditionedUNet(conv_dim=2, in_channels=3, out_channels=3,
                            stage_channels=[64, 128, 256, 512],
                            time_conditioning=True)
    out = model(x, time=t)

    # Flow-matching velocity field on point sets
    model = make_conditioned_point_to_point_model(
        point_dim=3, output_dim=3, embedding_dim=128,
        depth=6, num_heads=4, time_conditioning=True)
    out = model(x, time=t)
    ```

    See: [U-Net](api/unet.md) · [Transformer presets](api/transformer/presets.md)

=== "I want a patch / ViT model"

    **PatchTransformerND** handles 1-D, 2-D, and 3-D grids and supports grid reconstruction, token-level output, or a pooled vector.

    ```python
    from ml_suite.models.transformer.presets import make_patch_grid_model, make_patch_classifier

    # ViT-style image-to-image (e.g. segmentation backbone)
    model = make_patch_grid_model(input_dim=2, in_channels=3, out_channels=1,
                                  patch_size=16, embedding_dim=256,
                                  depth=12, num_heads=8)

    # ViT classifier
    model = make_patch_classifier(input_dim=2, in_channels=3, num_classes=1000,
                                  patch_size=16, embedding_dim=768,
                                  depth=12, num_heads=12)
    ```

    See: [Transformer presets](api/transformer/presets.md) · [Transformer models](api/transformer/models.md)

=== "I want to build something custom"

    Compose lower-level primitives directly.

    ```python
    from ml_suite.models.transformer import (
        ContinuousInputTokenizer, TransformerStack,
        PooledHead, RotaryEmbedding,
    )
    from ml_suite.models.linear import MLP

    tokenizer = ContinuousInputTokenizer(input_dim=64, embedding_dim=256)
    stack = TransformerStack(embedding_dim=256, depth=6, num_heads=8)
    head = PooledHead(embedding_dim=256, output_dim=10)
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
