"""Input tokenizers for continuous, discrete, set, and patch-structured data."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .heads import make_head_mlp
from .utils import compute_patch_grid, ensure_tuple


class ContinuousInputTokenizer(nn.Module):
    """Project continuous token features into transformer embedding space."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_layers: int = 1,
        hidden_dim: int | None = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.projection = make_head_mlp(
            input_dim=input_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim if hidden_dim is not None else embedding_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a continuous tensor of shape (batch, tokens, input_dim)."""
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, tokens, input_dim). Got {x.shape}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}. Got {x.shape[-1]}.")
        return self.projection(x)


class SetTokenizer(ContinuousInputTokenizer):
    """Tokenizer for unordered set-like inputs."""


class DiscreteTokenTokenizer(nn.Module):
    """Embed integer token IDs."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive. Got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs shaped (batch, tokens)."""
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise ValueError(
                f"token_ids must be in [0, {self.vocab_size}). "
                f"Got range [{token_ids.min()}, {token_ids.max()}]."
            )
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape (batch, tokens). Got {token_ids.shape}.")
        return self.embedding(token_ids.long())


class PatchTokenizerND(nn.Module):
    """Patchify 1D, 2D, or 3D grid data into token embeddings."""

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        embedding_dim: int,
        patch_size: int | Sequence[int],
    ) -> None:
        super().__init__()

        if spatial_dim not in (1, 2, 3):
            raise ValueError(f"spatial_dim must be 1, 2, or 3. Got {spatial_dim}.")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive. Got {in_channels}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.patch_size = ensure_tuple(patch_size, spatial_dim, "patch_size")

        conv_cls = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }[spatial_dim]

        self.projection = conv_cls(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Return patch tokens and the flattened patch-grid shape."""
        expected_ndim = self.spatial_dim + 2
        if x.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D input. Got {x.shape}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels. Got {x.shape[1]}.")

        grid_shape = compute_patch_grid(x.shape[2:], self.patch_size)
        patches = self.projection(x)

        tokens = patches.flatten(start_dim=2).transpose(1, 2)
        return tokens, grid_shape
