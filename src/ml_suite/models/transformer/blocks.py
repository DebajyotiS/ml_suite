"""Transformer feed-forward and residual attention blocks."""

from __future__ import annotations

import torch
from torch import nn

from ml_suite.models.transformer.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from ml_suite.models.transformer.types import NormType, PositionalEncodingMode
from ml_suite.utils.activations import get_activation


def build_norm(norm_type: NormType, embedding_dim: int) -> nn.Module:
    """Instantiate a normalisation layer by type name.

    Args:
        norm_type: 'layer' for LayerNorm or 'rms' for RMSNorm.
        embedding_dim: Feature dimension to normalise over.

    Returns:
        The requested normalisation module.

    Raises:
        ValueError: If norm_type is not supported.
    """
    if norm_type == "layer":
        return nn.LayerNorm(embedding_dim)
    if norm_type == "rms":
        return nn.RMSNorm(embedding_dim)  # fused kernel, same interface
    raise ValueError(f"Unsupported norm_type: {norm_type}.")


class FeedForward(nn.Module):
    """Transformer feed-forward network."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive. Got {hidden_dim}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer feed-forward network to tokens.

        Args:
            x: Token tensor of shape (batch, tokens, embedding_dim).

        Returns:
            Output tensor of the same shape.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional cross-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
        use_cross_attention: bool = False,
        cross_attention_dim: int | None = None,
        norm_type: NormType = "layer",
        positional_encoding: PositionalEncodingMode = "none",
    ) -> None:
        super().__init__()

        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive. Got {mlp_ratio}.")
        if use_cross_attention and cross_attention_dim is None:
            raise ValueError("cross_attention_dim is required when use_cross_attention=True.")
        if use_cross_attention and cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive. Got {cross_attention_dim}.")

        self.embedding_dim = embedding_dim
        self.use_cross_attention = use_cross_attention

        self.self_norm = build_norm(norm_type, embedding_dim)
        self.self_attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            causal=causal,
            positional_encoding=positional_encoding,
        )

        if use_cross_attention:
            self.cross_norm = build_norm(norm_type, embedding_dim)
            self.cross_attention = MultiHeadCrossAttention(
                query_dim=embedding_dim,
                context_dim=cross_attention_dim,  # type: ignore
                num_heads=num_heads,
                dropout=attention_dropout,
            )

        self.ff_norm = build_norm(norm_type, embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=int(mlp_ratio * embedding_dim),
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply pre-norm self-attention, optional cross-attention, and feed-forward with residuals.

        Args:
            x: Token tensor of shape (batch, tokens, embedding_dim).
            mask: Optional boolean self-attention mask of shape (batch, tokens). True = valid token.
            context: Cross-attention context of shape (batch, context_tokens, cross_attention_dim).
                Required when use_cross_attention=True.
            context_mask: Optional boolean mask of shape (batch, context_tokens). True = valid token.

        Returns:
            Output tensor of shape (batch, tokens, embedding_dim).
        """
        x = x + self.self_attention(self.self_norm(x), mask=mask)

        if self.use_cross_attention:
            if context is None:
                raise ValueError("context is required when `use_cross_attention`=True.")
            x = x + self.cross_attention(
                self.cross_norm(x),
                context=context,
                context_mask=context_mask,
            )
        elif context is not None:
            raise ValueError("context was provided, but `use_cross_attention`=False.")

        x = x + self.feed_forward(self.ff_norm(x))
        return x
