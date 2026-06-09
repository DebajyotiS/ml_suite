"""Transformer block stacks."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import TransformerBlock, build_norm
from .types import NormType, PositionalEncodingMode


class TransformerStack(nn.Module):
    """Stack of pre-norm transformer blocks."""

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
        use_cross_attention: bool = False,
        cross_attention_dim: int | None = None,
        norm_type: NormType = "layer",
        positional_encoding: PositionalEncodingMode = "none",
        final_norm: bool = True,
    ) -> None:
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if use_cross_attention and cross_attention_dim is None:
            raise ValueError("cross_attention_dim is required when use_cross_attention=True.")
        self.embedding_dim = embedding_dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    causal=causal,
                    use_cross_attention=use_cross_attention,
                    cross_attention_dim=cross_attention_dim,
                    norm_type=norm_type,
                    positional_encoding=positional_encoding,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = build_norm(norm_type, embedding_dim) if final_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply all transformer blocks to a token tensor."""
        for block in self.blocks:
            x = block(
                x,
                mask=mask,
                context=context,
                context_mask=context_mask,
            )
        return self.final_norm(x)
