"""Token decoders for per-token, grid, and query-set outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .heads import TokenwiseHead
from .utils import ensure_tuple, num_tokens_from_grid


class TokenDecoder(TokenwiseHead):
    """Decode each token independently."""


class PatchDecoderND(nn.Module):
    """Decode patch tokens back to a 1D, 2D, or 3D grid."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        out_channels: int,
        patch_size: int | Sequence[int],
    ) -> None:
        super().__init__()

        if input_dim not in (1, 2, 3):
            raise ValueError(f"input_dim must be 1, 2, or 3. Got {input_dim}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive. Got {out_channels}.")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.patch_size = ensure_tuple(patch_size, input_dim, "patch_size")

        conv_transpose_cls = {
            1: nn.ConvTranspose1d,
            2: nn.ConvTranspose2d,
            3: nn.ConvTranspose3d,
        }[input_dim]

        self.projection = conv_transpose_cls(
            in_channels=embedding_dim,
            out_channels=out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        grid_shape: Sequence[int],
    ) -> torch.Tensor:
        """Decode tokens shaped (batch, patches, embedding_dim) to a grid."""
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (batch, patches, dim). Got {tokens.shape}.")
        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}. Got {tokens.shape[-1]}."
            )

        grid_shape = tuple(grid_shape)
        if len(grid_shape) != self.input_dim:
            raise ValueError(f"grid_shape must have length {self.input_dim}. Got {grid_shape}.")
        if tokens.shape[1] != num_tokens_from_grid(grid_shape):
            raise ValueError(
                f"Number of tokens {tokens.shape[1]} does not match grid shape {grid_shape}."
            )

        batch = tokens.shape[0]
        x = tokens.transpose(1, 2).contiguous().view(batch, self.embedding_dim, *grid_shape)
        return self.projection(x)


class QuerySetDecoder(nn.Module):
    """Decode a global latent vector into a fixed-size set using learned queries."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        num_queries: int,
        query_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive. Got {latent_dim}.")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive. Got {output_dim}.")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive. Got {num_queries}.")
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive. Got {query_dim}.")

        self.num_queries = num_queries
        self.query_dim = query_dim
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.latent_projection = nn.Linear(latent_dim, query_dim)
        self.head = TokenwiseHead(
            embedding_dim=query_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector into a fixed-size set of output tokens.

        Learned query vectors are biased by a projection of the latent before being mapped
        through the token-wise head to produce the output set.

        Args:
            latent: Global latent vector of shape (batch, latent_dim).

        Returns:
            Output token set of shape (batch, num_queries, output_dim).
        """
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape (batch, latent_dim). Got {latent.shape}.")

        batch = latent.shape[0]
        query_bias = self.latent_projection(latent)[:, None, :]
        queries = self.queries[None, :, :].expand(batch, -1, -1)
        return self.head(queries + query_bias)
