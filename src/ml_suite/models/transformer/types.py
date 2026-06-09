"""Shared type aliases for transformer modules."""

from typing import Literal

PositionalEncodingMode = Literal["none", "nope", "learned", "sinusoidal", "rope"]
PoolingMode = Literal["mean", "max", "cls", "last", "attention"]
NormType = Literal["layer", "rms"]
AttentionType = Literal["self", "cross", "self_cross"]
TimeEmbeddingType = Literal["sinusoidal", "learned"]
PatchOutputMode = Literal["grid", "tokens", "vector"]
