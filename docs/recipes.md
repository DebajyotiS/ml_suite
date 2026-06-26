# Recipes

Short, working code patterns for common tasks. No parameter explanations here. If you need those, see the [API reference](api/linear.md).

---

## Training a flow-matching model on images

Flow matching trains a model to predict the velocity field that moves noise toward data along a straight path. The loss is the MSE between the predicted velocity and the target velocity at a random interpolation time.

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
```

---

## Adding multiple conditioning signals

Pass `time_conditioning=True` and `num_classes=N` at construction time, then pass both `time` and `class_labels` at every forward call.

```python
import torch
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
```

---

## Building a custom architecture from primitives

The standard pattern is tokenizer → transformer stack → head. Use this when no preset fits your architecture.

```python
import torch
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
```
