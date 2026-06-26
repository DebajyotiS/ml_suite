# Recipes

Each recipe is a complete, runnable code block. Paste it into a notebook or script and run it directly. For parameter details, see the [API reference](api/linear.md). Recipes are organised by task, not by module.

| Task | Recipe |
|---|---|
| Train a generative model on images | [Flow matching on images](#training-a-flow-matching-model-on-images) |
| Same model, different data dimensionality | [1D and 2D with the same model](#using-the-same-model-on-1d-and-2d-data) |
| Condition on time and class label simultaneously | [Multiple conditioning signals](#adding-multiple-conditioning-signals) |
| Build a non-standard architecture | [Custom architecture from primitives](#building-a-custom-architecture-from-primitives) |
| Classify images | [Image classification with ConvNet](#classifying-images-with-convnet) |
| Extract spatial features from images | [Feature extraction with ConvNet](#feature-extraction-with-convnet) |
| Classify 3D point clouds | [Point cloud classification](#classifying-point-clouds) |
| Flow matching on point clouds | [Flow matching on point clouds](#flow-matching-on-point-clouds) |
| Classify variable-length sequences with padding | [Sequence classification with padding masks](#sequence-classification-with-padding-masks) |
| Segment or reconstruct spatial grids with a ViT | [Patch transformer for grid reconstruction](#patch-transformer-for-grid-reconstruction) |
| Condition generation on a token sequence | [Cross-attention conditioning](#conditioning-generation-on-a-token-sequence) |
| Build a conditioned MLP | [FiLM-conditioned MLP](#film-conditioned-mlp) |
| Debug receptive field | [Inspecting receptive field](#checking-the-receptive-field-of-a-convnet) |

---

## Training a flow-matching model on images

We pick a random time `t` in [0, 1], mix the data sample with noise at that fraction, then ask the model to predict the direction from noise toward the data sample. That direction is the velocity target. The model learns to point from noise to data at every point along the interpolation path. At inference time you integrate this velocity field to move noise to data.

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

batch_size = 4
x = torch.randn(batch_size, 3, 64, 64)   # real data sample
noise = torch.randn_like(x)
t = torch.rand(batch_size)               # random time in [0, 1]
t_view = t.view(-1, 1, 1, 1)

x_t = (1 - t_view) * noise + t_view * x  # linear interpolation
velocity_target = x - noise               # constant velocity field

velocity_pred = model(x_t, time=t)
loss = F.mse_loss(velocity_pred, velocity_target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())
```

---

## Using the same model on 1D and 2D data

The only change between a 1D and 2D model is `conv_dim`. Everything else, including conditioning, training loop, and loss, stays the same.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.unet import ConditionedUNet

# 1D: audio or time-series denoiser
model_1d = ConditionedUNet(
    conv_dim=1,
    in_channels=1,
    out_channels=1,
    stage_channels=[32, 64, 128],
    time_conditioning=True,
)

# 2D: image denoiser. Only conv_dim changes.
model_2d = ConditionedUNet(
    conv_dim=2,
    in_channels=1,
    out_channels=1,
    stage_channels=[32, 64, 128],
    time_conditioning=True,
)

t = torch.rand(4)

signal = torch.randn(4, 1, 1024)
out_1d = model_1d(signal, time=t)   # (4, 1, 1024)

image = torch.randn(4, 1, 64, 64)
out_2d = model_2d(image, time=t)    # (4, 1, 64, 64)

target_1d = torch.randn_like(out_1d)
target_2d = torch.randn_like(out_2d)
print(F.mse_loss(out_1d, target_1d).item())   # ~1.0 for random data
print(F.mse_loss(out_2d, target_2d).item())   # ~1.0 for random data
```

---

## Adding multiple conditioning signals

Pass `time_conditioning=True` and `num_classes=N` at construction time, then pass both `time` and `class_labels` at every forward call.

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
    num_classes=10,
    attention_downsample_factors=[4, 8],
)

x = torch.randn(4, 3, 64, 64)
t = torch.rand(4)
y = torch.randint(0, 10, (4,))

out = model(x, time=t, class_labels=y)   # (4, 3, 64, 64)

target = torch.randn_like(out)
print(F.mse_loss(out, target).item())   # ~1.0 for random data
```

---

## Building a custom architecture from primitives

Use this pattern when no preset matches your architecture. The three pieces connect the same way in every custom model: tokenizer maps raw input to embedding space, the stack processes it, and the head maps it to your output.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer import (
    ContinuousInputTokenizer,
    TransformerStack,
    PooledHead,
)

# Step 1: project raw features into the transformer's embedding space
tokenizer = ContinuousInputTokenizer(input_dim=64, embedding_dim=256)

# Step 2: apply the transformer layers
stack = TransformerStack(embedding_dim=256, depth=6, num_heads=8)

# Step 3: collapse the token sequence to a single vector for the task head
head = PooledHead(embedding_dim=256, output_dim=10)

x = torch.randn(8, 32, 64)   # (batch, tokens, features)
tokens = tokenizer(x)         # (8, 32, 256)
encoded = stack(tokens)       # (8, 32, 256)
out = head(encoded)           # (8, 10)

target = torch.randint(0, 10, (8,))
loss = F.cross_entropy(out, target)
loss.backward()
print(loss.item())   # ~2.3 for a random init with 10 classes
```

---

## Classifying images with ConvNet

`ConvNet` is a multi-stage convolutional classifier. Set `num_classes` and it adds global pooling and a linear head. The same class works for 1D signals, 2D images, and 3D volumes by changing only `conv_dim`.

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(8, 3, 64, 64)
labels = torch.randint(0, 10, (8,))

logits = model(x)                              # (8, 10)
loss = F.cross_entropy(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())                             # ~2.3 for a random init with 10 classes
```

To classify 1D signals instead, set `conv_dim=1` and change the input to `(batch, channels, length)`. To classify 3D volumes, set `conv_dim=3` and use a 5D input tensor. No other changes are needed.

---

## Feature extraction with ConvNet

Omit `num_classes` and `ConvNet` returns the final-stage spatial feature map instead of class logits. Use this when you want a convolutional encoder inside a larger model.

```python
import torch
from ml_suite.models.convolution import ConvNet

backbone = ConvNet(
    conv_dim=2,
    in_channels=3,
    stage_channels=[64, 128, 256],
    blocks_per_stage=2,
    # no num_classes -- returns feature map
)

x = torch.randn(4, 3, 64, 64)
features = backbone(x)
print(features.shape)   # (4, 256, 8, 8) -- spatial dims shrink with each stage
```

The spatial dimensions shrink by a factor of 2 at each stage because the default `downsample_mode='stride'` uses strided convolutions. With 3 stages on a 64x64 input you get 8x8 feature maps. You can feed these into any downstream head.

---

## Classifying point clouds

Each point in the cloud becomes a token. The transformer processes all points jointly and pools them to a single class prediction. Because no positional encoding is applied, the model is invariant to the order of points in the input.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer.presets import make_point_cloud_classifier

model = make_point_cloud_classifier(
    point_dim=3,        # XYZ coordinates
    num_classes=40,
    embedding_dim=256,
    depth=6,
    num_heads=4,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(8, 1024, 3)       # (batch, points, xyz)
labels = torch.randint(0, 40, (8,))

logits = model(x)                  # (8, 40)
loss = F.cross_entropy(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())
```

If each point also has per-point features (e.g. normals or intensity), pass them concatenated to the coordinates and set `feature_dim` to the number of extra features. The model receives `point_dim + feature_dim` as its total input width.

---

## Flow matching on point clouds

The flow-matching objective is identical to the image recipe. The only change is the model. `make_conditioned_point_to_point_model` produces a velocity vector per input point instead of a velocity map per spatial location.

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

x = torch.randn(4, 1024, 3)       # (batch, points, xyz)
noise = torch.randn_like(x)
t = torch.rand(4)
t_view = t.view(-1, 1, 1)

x_t = (1 - t_view) * noise + t_view * x
velocity_target = x - noise

velocity_pred = model(x_t, time=t)   # (4, 1024, 3)
loss = F.mse_loss(velocity_pred, velocity_target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())
```

The interpolation and loss are exactly the same as the image recipe. The model sees each point as a token and predicts a velocity vector for each one independently.

---

## Sequence classification with padding masks

When sequences in a batch have different lengths, pad them to the same length and pass a boolean mask. Positions where the mask is `False` are ignored in attention and pooling, so padding does not affect predictions.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer.presets import make_sequence_classifier

model = make_sequence_classifier(
    input_dim=32,
    num_classes=5,
    embedding_dim=128,
    depth=4,
    num_heads=4,
    max_length=256,
)

# three sequences with lengths 80, 120, and 60, padded to 120
x = torch.randn(3, 120, 32)
lengths = torch.tensor([80, 120, 60])

# True = valid token, False = padding
mask = torch.arange(120).unsqueeze(0) < lengths.unsqueeze(1)   # (3, 120)

logits = model(x, mask=mask)           # (3, 5)
labels = torch.tensor([0, 2, 4])
loss = F.cross_entropy(logits, labels)
loss.backward()
print(loss.item())
```

The mask shape must be `(batch, tokens)` with dtype `bool`. A `True` entry means the token is real; `False` means it is padding and will be masked from attention. If all sequences in a batch are the same length, you can omit the mask entirely.

---

## Patch transformer for grid reconstruction

The input grid is split into non-overlapping patches, each treated as one token. After the transformer processes the tokens, they are reassembled into a spatial grid at the original resolution. Use this as a segmentation backbone or a denoising head on spatial data.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.transformer.presets import make_patch_grid_model

model = make_patch_grid_model(
    input_dim=2,
    in_channels=3,
    out_channels=1,           # binary segmentation mask
    patch_size=16,
    embedding_dim=256,
    depth=6,
    num_heads=8,
)

# spatial dims must be divisible by patch_size
x = torch.randn(2, 3, 256, 256)
pred = model(x)                   # (2, 1, 256, 256)

target = torch.randint(0, 2, (2, 1, 256, 256)).float()
loss = F.binary_cross_entropy_with_logits(pred, target)
loss.backward()
print(loss.item())
```

The spatial dimensions of your input must be divisible by `patch_size`. For a 256x256 image with `patch_size=16`, the model produces 256 patch tokens. If your input is not divisible, either resize it or use a smaller `patch_size`. This works for 1D sequences and 3D volumes too by changing `input_dim`.

---

## Conditioning generation on a token sequence

Set `attention_type='self_cross'` and `cross_attention_dim` to match your context embedding size. Pass `cross_context` at every forward call. The model attends to the context tokens at each spatial attention block.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.unet import ConditionedUNet

model = ConditionedUNet(
    conv_dim=2,
    in_channels=3,
    out_channels=3,
    stage_channels=[64, 128, 256],
    time_conditioning=True,
    attention_downsample_factors=[4],
    attention_type="self_cross",
    cross_attention_dim=512,          # must match context embedding size
)

x = torch.randn(2, 3, 64, 64)
t = torch.rand(2)
context = torch.randn(2, 16, 512)    # (batch, context_tokens, context_dim)

out = model(x, time=t, cross_context=context)   # (2, 3, 64, 64)

target = torch.randn_like(out)
loss = F.mse_loss(out, target)
loss.backward()
print(loss.item())
```

The context tensor can come from any encoder: a text encoder, a graph encoder, another modality. Its token dimension can be any length. If your context sequences have variable lengths, pass `cross_context_mask` as a boolean tensor of shape `(batch, context_tokens)` with `True` for valid tokens.

---

## FiLM-conditioned MLP

`MLP` with `context_injection='film'` applies a learned scale and shift from a conditioning vector at every layer. This is the standard mechanism diffusion models use to inject timestep information, now available as a standalone MLP stack.

```python
import torch
import torch.nn.functional as F
from ml_suite.models.linear import MLP

model = MLP(
    input_dim=64,
    hidden_dim=64,
    num_layers=4,
    context_dim=16,
    context_injection="film",
    do_residual=True,
)

x = torch.randn(8, 64)
condition = torch.randn(8, 16)    # e.g. a time embedding or class embedding

out = model(x, context=condition)   # (8, 64)

target = torch.randn_like(out)
loss = F.mse_loss(out, target)
loss.backward()
print(loss.item())
```

The `do_residual=True` setting adds skip connections between layers where input and output dimensions match. For a diffusion score network on flat data, replace `condition` with a sinusoidal time embedding from `ml_suite.utils.conditioning.TimeEmbeddingMLP`.

---

## Checking the receptive field of a ConvNet

Call `model.print_receptive_field()` after construction to see how many input pixels each output feature covers after each layer. If the receptive field is smaller than the spatial patterns you care about, add more stages or increase `blocks_per_stage`.

```python
from ml_suite.models.convolution import ConvNet

model = ConvNet(
    conv_dim=2,
    in_channels=3,
    stage_channels=[64, 128, 256, 512],
    blocks_per_stage=2,
)

model.print_receptive_field()
# prints a table: Layer | Layer RF | Cumulative RF | Cumulative Stride
```

The cumulative stride tells you how much the spatial resolution has been reduced at each layer. For a 512x512 input with 4 stages, the final feature map covers the full input in its receptive field, which is typically what you want for whole-image classification. For tasks requiring fine spatial detail, use fewer stages or add a U-Net decoder.
