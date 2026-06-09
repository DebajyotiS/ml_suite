"""Shared type aliases for U-Net modules."""

from typing import Literal

NormType = Literal["batch", "group", "layer"] | None
DownsampleMode = Literal["stride", "pool"]
UpsampleMode = Literal["interpolate", "transpose"]
SkipMode = Literal["concat", "add"]
ShapePolicy = Literal["resize", "error"]
AttentionLocation = Literal["encoder", "decoder", "both", "bottleneck"]
AttentionType = Literal["self", "cross", "self_cross"]
TimeEmbeddingType = Literal["sinusoidal", "learned"]
