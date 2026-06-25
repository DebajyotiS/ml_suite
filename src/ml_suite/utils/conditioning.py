"""Shared time embedding primitives used across UNet and Transformer conditioning."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

TimeEmbeddingType = Literal["sinusoidal", "learned"]


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar time values of shape (B,) or (B, 1) to sinusoidal embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        exponent = -math.log(10_000.0) * torch.arange(half_dim) / max(half_dim - 1, 1)
        self.register_buffer("frequencies", torch.exp(exponent))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed scalar time values into a sinusoidal representation.

        Args:
            time: Timestep tensor of shape (batch,) or (batch, 1).

        Returns:
            Sinusoidal embeddings of shape (batch, embedding_dim).
        """
        if time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        if time.ndim != 1:
            raise ValueError(f"time must have shape (batch,) or (batch, 1). Got {time.shape}.")
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return time[:, None].float()
        args = time.float()[:, None] * self.frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeEmbeddingMLP(nn.Module):
    """Project scalar time values to a target embedding dimension."""

    def __init__(
        self,
        embedding_dim: int,
        embedding_type: TimeEmbeddingType = "sinusoidal",
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if embedding_type not in ("sinusoidal", "learned"):
            raise ValueError(f"Unsupported embedding_type: {embedding_type}.")
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        if embedding_type == "sinusoidal":
            self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
            input_dim = embedding_dim
        else:
            input_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_mult * embedding_dim),
            nn.SiLU(),
            nn.Linear(hidden_mult * embedding_dim, embedding_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Map scalar time values to a target-dimensioned embedding vector.

        Args:
            time: Timestep tensor of shape (batch,) or (batch, 1).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        if self.embedding_type == "sinusoidal":
            emb = self.time_embedding(time)
        else:
            if time.ndim == 1:
                emb = time[:, None].float()
            elif time.ndim == 2 and time.shape[1] == 1:
                emb = time.float()
            else:
                raise ValueError(f"time must have shape (batch,) or (batch, 1). Got {time.shape}.")
        return self.mlp(emb)
