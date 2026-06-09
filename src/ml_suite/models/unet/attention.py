"""Spatial attention blocks for dimension-agnostic U-Nets."""

import torch
from torch import nn

from .types import AttentionType


class SpatialAttentionBlock(nn.Module):
    """Spatial attention supporting self-attention and cross-attention.

    A convolutional feature map ``(B, C, *spatial)`` is flattened to spatial
    tokens ``(B, N, C)``, processed with attention, then reshaped back.
    """

    def __init__(
        self,
        channels: int,
        attention_type: AttentionType = "self",
        num_heads: int = 4,
        head_dim: int | None = None,
        cross_attention_dim: int | None = None,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive. Got {channels}.")
        if attention_type not in ("self", "cross", "self_cross"):
            raise ValueError(
                f"attention_type must be 'self', 'cross', or 'self_cross'. Got {attention_type}."
            )
        if attention_type in ("cross", "self_cross") and cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim is required when attention_type is 'cross' or 'self_cross'."
            )
        if cross_attention_dim is not None and cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive. Got {cross_attention_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive. Got {num_heads}.")
        if head_dim is None:
            if channels % num_heads != 0:
                raise ValueError(
                    f"channels ({channels}) must be divisible by num_heads ({num_heads}) "
                    "when head_dim is None."
                )
            inner_dim = channels
        else:
            if head_dim <= 0:
                raise ValueError(f"head_dim must be positive. Got {head_dim}.")
            inner_dim = num_heads * head_dim
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")

        self.channels = channels
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.cross_attention_dim = cross_attention_dim
        self.uses_self_attention = attention_type in ("self", "self_cross")
        self.uses_cross_attention = attention_type in ("cross", "self_cross")

        if self.uses_self_attention:
            self.self_norm = nn.LayerNorm(channels)
            self.self_input_projection = (
                nn.Identity() if inner_dim == channels else nn.Linear(channels, inner_dim)
            )
            self.self_attention = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.self_output_projection = (
                nn.Identity() if inner_dim == channels else nn.Linear(inner_dim, channels)
            )
        else:
            self.self_norm = None
            self.self_input_projection = None
            self.self_attention = None
            self.self_output_projection = None

        if self.uses_cross_attention:
            assert cross_attention_dim is not None
            self.cross_norm = nn.LayerNorm(channels)
            self.cross_query_projection = (
                nn.Identity() if inner_dim == channels else nn.Linear(channels, inner_dim)
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                kdim=cross_attention_dim,
                vdim=cross_attention_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_output_projection = (
                nn.Identity() if inner_dim == channels else nn.Linear(inner_dim, channels)
            )
        else:
            self.cross_norm = None
            self.cross_query_projection = None
            self.cross_attention = None
            self.cross_output_projection = None

        self.ff_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, ff_mult * channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * channels, channels),
            nn.Dropout(dropout),
        )

    def _get_key_padding_mask(
        self,
        tokens: torch.Tensor,
        cross_context: torch.Tensor | None,
        cross_context_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.uses_cross_attention:
            if cross_context is not None:
                raise ValueError(
                    "cross_context was provided, but this block does not use cross-attention."
                )
            return None

        if cross_context is None:
            raise ValueError("cross_context must be provided for cross-attention.")
        if cross_context.ndim != 3:
            raise ValueError(
                f"cross_context must have shape (batch, tokens, dim). Got {cross_context.shape}."
            )
        if cross_context.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"cross_context batch size ({cross_context.shape[0]}) must match "
                f"input batch size ({tokens.shape[0]})."
            )
        if cross_context.shape[2] != self.cross_attention_dim:
            raise ValueError(
                f"cross_context feature dimension must be {self.cross_attention_dim}. "
                f"Got {cross_context.shape[2]}."
            )
        if cross_context_mask is None:
            return None
        if cross_context_mask.shape != cross_context.shape[:2]:
            raise ValueError(
                f"cross_context_mask must have shape {cross_context.shape[:2]}. "
                f"Got {cross_context_mask.shape}."
            )
        if cross_context_mask.dtype != torch.bool:
            raise ValueError("cross_context_mask must be a boolean tensor.")

        # PyTorch expects True for tokens to ignore. User-facing mask uses True=valid.
        return ~cross_context_mask

    def forward(
        self,
        x: torch.Tensor,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError("Expected x with shape (B, C, *spatial_dims).")

        batch_size, channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, but got {channels}.")

        tokens = x.flatten(start_dim=2).transpose(1, 2)
        key_padding_mask = self._get_key_padding_mask(tokens, cross_context, cross_context_mask)

        if self.uses_self_attention:
            assert self.self_norm is not None
            assert self.self_input_projection is not None
            assert self.self_attention is not None
            assert self.self_output_projection is not None
            residual = tokens
            h = self.self_input_projection(self.self_norm(tokens))
            h, _ = self.self_attention(h, h, h, need_weights=False)
            tokens = residual + self.self_output_projection(h)

        if self.uses_cross_attention:
            assert cross_context is not None
            assert self.cross_norm is not None
            assert self.cross_query_projection is not None
            assert self.cross_attention is not None
            assert self.cross_output_projection is not None
            residual = tokens
            h = self.cross_query_projection(self.cross_norm(tokens))
            h, _ = self.cross_attention(
                h,
                cross_context,
                cross_context,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            tokens = residual + self.cross_output_projection(h)

        residual = tokens
        tokens = residual + self.feed_forward(self.ff_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)
