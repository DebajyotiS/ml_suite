"""Convenience constructors for common data layouts."""

from __future__ import annotations

from .models import (
    ConditionedTokenTransformer,
    PatchTransformerND,
    TokenToClassTransformer,
    TokenToTokenTransformer,
)
from .types import PoolingMode


def make_point_cloud_classifier(
    point_dim: int,
    num_classes: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    feature_dim: int = 0,
    pooling: PoolingMode = "mean",
) -> TokenToClassTransformer:
    """Create a permutation-invariant point-cloud classifier."""
    return TokenToClassTransformer(
        input_dim=point_dim + feature_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        pooling=pooling,
        positional_encoding="none",
    )


def make_point_to_point_model(
    point_dim: int,
    output_dim: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    feature_dim: int = 0,
) -> TokenToTokenTransformer:
    """Create a permutation-equivariant point-to-point model."""
    return TokenToTokenTransformer(
        input_dim=point_dim + feature_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        positional_encoding="none",
    )


def make_conditioned_point_to_point_model(
    point_dim: int,
    output_dim: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    feature_dim: int = 0,
    time_conditioning: bool = False,
    global_context_dim: int | None = None,
    cross_attention_dim: int | None = None,
) -> ConditionedTokenTransformer:
    """Create a conditional point-to-point model."""
    return ConditionedTokenTransformer(
        input_dim=point_dim + feature_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        positional_encoding="none",
        time_conditioning=time_conditioning,
        global_context_dim=global_context_dim,
        cross_attention_dim=cross_attention_dim,
    )


def make_sequence_classifier(
    input_dim: int,
    num_classes: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    max_length: int,
    pooling: PoolingMode = "mean",
) -> TokenToClassTransformer:
    """Create an ordered sequence classifier."""
    return TokenToClassTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        pooling=pooling,
        max_length=max_length,
        positional_encoding="rope",
    )


def make_patch_grid_model(
    input_dim: int,
    in_channels: int,
    out_channels: int,
    patch_size: int | tuple[int, ...],
    embedding_dim: int,
    depth: int,
    num_heads: int,
) -> PatchTransformerND:
    """Create a grid-to-grid patch transformer."""
    return PatchTransformerND(
        input_dim=input_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        output_mode="grid",
        positional_encoding="sinusoidal",
    )


def make_patch_classifier(
    input_dim: int,
    in_channels: int,
    num_classes: int,
    patch_size: int | tuple[int, ...],
    embedding_dim: int,
    depth: int,
    num_heads: int,
) -> PatchTransformerND:
    """Create a patch-based classifier for 1D, 2D, or 3D grids."""
    return PatchTransformerND(
        input_dim=input_dim,
        in_channels=in_channels,
        out_channels=in_channels,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        output_mode="vector",
        vector_output_dim=num_classes,
        positional_encoding="sinusoidal",
    )
