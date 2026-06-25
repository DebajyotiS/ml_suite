"""Public token-centric transformer model compositions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .conditioning import TransformerConditioningBuilder
from .decoders import PatchDecoderND
from .heads import PooledHead, TokenwiseHead
from .positional import build_absolute_positional_embedding
from .stacks import TransformerStack
from .tokenization import ContinuousInputTokenizer, PatchTokenizerND
from .types import (
    PatchOutputMode,
    PoolingMode,
    PositionalEncodingMode,
    TimeEmbeddingType,
)
from .utils import normalize_positional_encoding_mode, validate_mask


def _stack_positional(mode: PositionalEncodingMode) -> PositionalEncodingMode:
    """RoPE is handled inside the stack; all other modes are applied externally."""
    return "rope" if mode == "rope" else "none"


class TokenToTokenTransformer(nn.Module):
    """Generic token-to-token transformer.

    Input:
        x.shape == (batch, tokens, input_dim)

    Output:
        out.shape == (batch, tokens, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        max_length: int | None = None,
        positional_encoding: PositionalEncodingMode = "learned",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        positional_encoding = normalize_positional_encoding_mode(positional_encoding)
        stack_positional = _stack_positional(positional_encoding)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.positional_encoding = positional_encoding

        self.tokenizer = ContinuousInputTokenizer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )
        self.absolute_position = build_absolute_positional_embedding(
            mode=positional_encoding,
            embedding_dim=embedding_dim,
            max_length=max_length,
        )
        self.stack = TransformerStack(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            positional_encoding=stack_positional,
        )
        self.head = TokenwiseHead(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map a token sequence to output tokens of the same length.

        Args:
            x: Input token tensor of shape (batch, tokens, input_dim).
            mask: Optional boolean valid-token mask of shape (batch, tokens). True = valid token.

        Returns:
            Output tensor of shape (batch, tokens, output_dim).
        """
        tokens = self.tokenizer(x)

        if mask is not None:
            validate_mask(mask, tokens)

        tokens = self.absolute_position(tokens)
        tokens = self.stack(tokens, mask=mask)
        return self.head(tokens)


class TokenToVectorTransformer(nn.Module):
    """Generic token-to-vector transformer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        pooling: PoolingMode = "mean",
        max_length: int | None = None,
        positional_encoding: PositionalEncodingMode = "learned",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        positional_encoding = normalize_positional_encoding_mode(positional_encoding)
        stack_positional = _stack_positional(positional_encoding)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        self.tokenizer = ContinuousInputTokenizer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )
        self.absolute_position = build_absolute_positional_embedding(
            mode=positional_encoding,
            embedding_dim=embedding_dim,
            max_length=max_length,
        )
        self.stack = TransformerStack(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            positional_encoding=stack_positional,
        )
        self.head = PooledHead(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            pooling=pooling,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map a token sequence to a single output vector.

        Args:
            x: Input token tensor of shape (batch, tokens, input_dim).
            mask: Optional boolean valid-token mask of shape (batch, tokens). True = valid token.

        Returns:
            Output vector of shape (batch, output_dim).
        """
        tokens = self.tokenizer(x)

        if mask is not None:
            validate_mask(mask, tokens)

        tokens = self.absolute_position(tokens)
        tokens = self.stack(tokens, mask=mask)
        return self.head(tokens, mask=mask)


class TokenToClassTransformer(TokenToVectorTransformer):
    """Generic token-to-class transformer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        pooling: PoolingMode = "mean",
        max_length: int | None = None,
        positional_encoding: PositionalEncodingMode = "learned",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive. Got {num_classes}.")

        super().__init__(
            input_dim=input_dim,
            output_dim=num_classes,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            pooling=pooling,
            max_length=max_length,
            positional_encoding=positional_encoding,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
        )


class ConditionedTokenTransformer(nn.Module):
    """Conditional token-to-token transformer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        max_length: int | None = None,
        positional_encoding: PositionalEncodingMode = "learned",
        time_conditioning: bool = False,
        time_embedding_type: TimeEmbeddingType = "sinusoidal",
        num_classes: int | None = None,
        class_dropout_prob: float = 0.0,
        global_context_dim: int | None = None,
        cross_attention_dim: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        positional_encoding = normalize_positional_encoding_mode(positional_encoding)
        stack_positional = _stack_positional(positional_encoding)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.cross_attention_dim = cross_attention_dim
        self.use_cross_attention = cross_attention_dim is not None

        self.tokenizer = ContinuousInputTokenizer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )
        self.absolute_position = build_absolute_positional_embedding(
            mode=positional_encoding,
            embedding_dim=embedding_dim,
            max_length=max_length,
        )
        self.conditioning = TransformerConditioningBuilder(
            embedding_dim=embedding_dim,
            time_conditioning=time_conditioning,
            time_embedding_type=time_embedding_type,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            global_context_dim=global_context_dim,
        )
        self.stack = TransformerStack(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            use_cross_attention=self.use_cross_attention,
            cross_attention_dim=cross_attention_dim,
            positional_encoding=stack_positional,
        )
        self.head = TokenwiseHead(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map a conditioned token sequence to output tokens of the same length.

        Global conditioning sources (time, class, context) are projected to embedding_dim, summed,
        and broadcast-added to all tokens before the transformer stack.

        Args:
            x: Input token tensor of shape (batch, tokens, input_dim).
            mask: Optional boolean valid-token mask of shape (batch, tokens). True = valid token.
            time: Diffusion timestep of shape (batch,). Required when time_conditioning=True.
            class_labels: Integer class indices of shape (batch,). Required when num_classes is set.
            global_context: Global context vector of shape (batch, global_context_dim).
                Required when global_context_dim is set.
            cross_context: Cross-attention token sequence of shape
                (batch, context_tokens, cross_attention_dim).
                Required when cross_attention_dim is set.
            cross_context_mask: Boolean mask of shape (batch, context_tokens). True = valid token.

        Returns:
            Output tensor of shape (batch, tokens, output_dim).
        """
        tokens = self.tokenizer(x)

        if mask is not None:
            validate_mask(mask, tokens)

        if self.use_cross_attention:
            if cross_context is None:
                raise ValueError("cross_context is required when cross_attention_dim is set.")
        elif cross_context is not None:
            raise ValueError("cross_context was provided, but cross_attention_dim is None.")

        condition = self.conditioning(
            batch_size=tokens.shape[0],
            device=tokens.device,
            dtype=tokens.dtype,
            time=time,
            class_labels=class_labels,
            global_context=global_context,
        )

        tokens = self.absolute_position(tokens)
        # TODO: Consider generalising to allow conditioning to be merged in other ways (e.g. concatenation, FiLM, etc.)
        tokens = tokens + condition[:, None, :]
        tokens = self.stack(
            tokens,
            mask=mask,
            context=cross_context,
            context_mask=cross_context_mask,
        )
        return self.head(tokens)


class PatchTransformerND(nn.Module):
    """Transformer over 1D, 2D, or 3D patches."""

    def __init__(
        self,
        input_dim: int,
        in_channels: int,
        out_channels: int | None,
        patch_size: int | Sequence[int],
        embedding_dim: int,
        depth: int,
        num_heads: int,
        output_mode: PatchOutputMode = "grid",
        vector_output_dim: int | None = None,
        max_length: int | None = None,
        positional_encoding: PositionalEncodingMode = "sinusoidal",
        pooling: PoolingMode = "mean",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if output_mode not in ("grid", "tokens", "vector"):
            raise ValueError(f"Unsupported output_mode: {output_mode}.")
        if output_mode == "vector" and vector_output_dim is None:
            raise ValueError("vector_output_dim is required when output_mode='vector'.")

        positional_encoding = normalize_positional_encoding_mode(positional_encoding)
        stack_positional = _stack_positional(positional_encoding)

        self.patch_decoder: PatchDecoderND | None = None
        self.vector_decoder: PooledHead | None = None

        self.output_mode = output_mode
        self.tokenizer = PatchTokenizerND(
            spatial_dim=input_dim,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
        )
        self.absolute_position = build_absolute_positional_embedding(
            mode=positional_encoding,
            embedding_dim=embedding_dim,
            max_length=max_length,
        )
        self.stack = TransformerStack(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            positional_encoding=stack_positional,
        )

        if output_mode == "grid":
            if out_channels is None:
                raise ValueError("out_channels is required when output_mode='grid'.")
            self.patch_decoder = PatchDecoderND(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                out_channels=out_channels,
                patch_size=patch_size,
            )
        elif output_mode == "vector":
            self.vector_decoder = PooledHead(
                embedding_dim=embedding_dim,
                output_dim=vector_output_dim,
                pooling=pooling,
            )
        else:  # "tokens"
            self.token_decoder: nn.Module = (
                TokenwiseHead(embedding_dim=embedding_dim, output_dim=out_channels)
                if out_channels is not None
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify the input grid, process with the transformer stack, and decode.

        Args:
            x: Spatial input of shape (batch, in_channels, *spatial). Each spatial dimension
                must be divisible by the corresponding patch_size entry.

        Returns:
            Decoded output whose shape depends on output_mode:
                - 'grid': (batch, out_channels, *spatial)
                - 'tokens': (batch, num_patches, out_channels) or
                    (batch, num_patches, embedding_dim) if out_channels is None
                - 'vector': (batch, vector_output_dim)
        """
        tokens, grid_shape = self.tokenizer(x)
        tokens = self.absolute_position(tokens)
        tokens = self.stack(tokens)

        if self.output_mode == "grid":
            return self.patch_decoder(tokens, grid_shape=grid_shape)
        if self.output_mode == "vector":
            return self.vector_decoder(tokens)
        return self.token_decoder(tokens)
