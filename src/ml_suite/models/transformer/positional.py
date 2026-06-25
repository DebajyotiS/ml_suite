"""Absolute and rotary positional encodings."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .types import PositionalEncodingMode
from .utils import normalize_positional_encoding_mode


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embedding for token tensors."""

    def __init__(self, max_length: int, embedding_dim: int) -> None:
        super().__init__()

        if max_length <= 0:
            raise ValueError(f"max_length must be positive. Got {max_length}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")

        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to a token tensor.

        Args:
            x: Token tensor of shape (batch, tokens, embedding_dim).

        Returns:
            Tensor of the same shape with positional embeddings added.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, tokens, dim). Got {x.shape}.")

        seq_len = x.shape[1]
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}.")

        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)[None, :, :].to(dtype=x.dtype)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed absolute sinusoidal positional embedding."""

    def __init__(self, embedding_dim: int, max_length: int | None = None) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        if max_length is not None and max_length <= 0:
            raise ValueError(f"max_length must be positive. Got {max_length}.")

    def _build(self, seq_len: int, device: torch.device) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        if half_dim == 0:
            return positions[:, None]

        exponent = -math.log(10_000.0) * torch.arange(
            half_dim,
            device=device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        args = positions[:, None] * frequencies[None, :]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add fixed sinusoidal positional embeddings to a token tensor.

        Args:
            x: Token tensor of shape (batch, tokens, embedding_dim).

        Returns:
            Tensor of the same shape with positional embeddings added.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, tokens, dim). Got {x.shape}.")

        seq_len = x.shape[1]
        if self.max_length is not None and seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}.")

        pos = self._build(seq_len, x.device).to(dtype=x.dtype)
        return x + pos[None, :, :]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache for attention Q/K tensors."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive. Got {head_dim}.")
        if base <= 0:
            raise ValueError(f"base must be positive. Got {base}.")

        self.head_dim = head_dim
        self.base = base

        rotary_dim = head_dim - (head_dim % 2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / max(rotary_dim, 1))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine rotation matrices for a given sequence length.

        Args:
            seq_len: Number of positions to generate embeddings for.
            device: Target device for the output tensors.
            dtype: Target dtype for the output tensors.

        Returns:
            A pair (cos, sin), each of shape (seq_len, rotary_dim // 2),
            where rotary_dim is head_dim rounded down to the nearest even number.
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a tensor shaped (batch, heads, tokens, head_dim)."""
    if x.ndim != 4:
        raise ValueError(f"x must have shape (batch, heads, tokens, head_dim). Got {x.shape}.")

    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    rotated = torch.stack(
        [
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ],
        dim=-1,
    ).flatten(start_dim=-2)

    if x_pass.numel() == 0:
        return rotated
    return torch.cat([rotated, x_pass], dim=-1)


def build_absolute_positional_embedding(
    mode: PositionalEncodingMode,
    embedding_dim: int,
    max_length: int | None,
) -> nn.Module:
    """Build an additive absolute positional embedding module."""
    mode = normalize_positional_encoding_mode(mode)

    if mode in ("none", "rope"):
        return nn.Identity()

    if mode == "learned":
        if max_length is None:
            raise ValueError("max_length is required for learned positional embeddings.")
        return LearnedPositionalEmbedding(max_length=max_length, embedding_dim=embedding_dim)

    if mode == "sinusoidal":
        return SinusoidalPositionalEmbedding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

    raise ValueError(f"Unsupported positional encoding mode: {mode}.")
