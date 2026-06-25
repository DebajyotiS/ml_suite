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
    """Create a permutation-invariant point-cloud classifier.

    No positional encoding is applied, so predictions are invariant to point ordering.
    Each point is treated as a token; global pooling aggregates them before classification.

    Args:
        point_dim: Spatial dimension of each point (e.g. 3 for XYZ coordinates).
        num_classes: Number of output classes.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        feature_dim: Optional per-point feature dimension appended to coordinates.
            The effective input dimension becomes point_dim + feature_dim.
        pooling: Pooling strategy applied over tokens before the classification head.

    Returns:
        A TokenToClassTransformer configured for point-cloud classification.
    """
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
    """Create a permutation-equivariant point-to-point model.

    No positional encoding is applied, so predictions are equivariant to point ordering.
    Each input point produces one output vector of output_dim features.

    Args:
        point_dim: Spatial dimension of each input point (e.g. 3 for XYZ).
        output_dim: Output feature dimension per point.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        feature_dim: Optional per-point feature dimension appended to coordinates.
            The effective input dimension becomes point_dim + feature_dim.

    Returns:
        A TokenToTokenTransformer configured for point-to-point prediction.
    """
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
    """Create a conditional point-to-point model.

    Suitable for score or velocity field estimation in diffusion and flow-matching models
    operating on point sets. No positional encoding is applied to the point tokens.

    Args:
        point_dim: Spatial dimension of each input point (e.g. 3 for XYZ).
        output_dim: Output feature dimension per point.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        feature_dim: Optional per-point feature dimension appended to coordinates.
            The effective input dimension becomes point_dim + feature_dim.
        time_conditioning: If True, accept a diffusion timestep tensor of shape (batch,).
        global_context_dim: Dimension of an optional global context vector.
        cross_attention_dim: If set, enable cross-attention with an external token sequence
            of this feature dimension.

    Returns:
        A ConditionedTokenTransformer configured for conditioned point-to-point prediction.
    """
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
    """Create an ordered sequence classifier.

    Uses RoPE positional encoding so position information is injected inside the attention
    mechanism rather than as an additive embedding. Suitable for variable-length sequences.

    Args:
        input_dim: Feature dimension of each input token.
        num_classes: Number of output classes.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        max_length: Maximum sequence length. Used to bound RoPE frequency computations.
        pooling: Pooling strategy applied over tokens before the classification head.

    Returns:
        A TokenToClassTransformer configured for sequence classification with RoPE.
    """
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
    """Create a grid-to-grid patch transformer.

    The input dense spatial grid is split into non-overlapping patches, each embedded as a token.
    After transformer processing, patch tokens are decoded back to a dense spatial grid of the
    same resolution. Sinusoidal positional encoding is applied to the patch tokens.

    Args:
        input_dim: Number of spatial dimensions (1, 2, or 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels in the reconstructed grid.
        patch_size: Patch size per spatial dimension. A single int is broadcast to all dimensions.
            The spatial dimensions of the input must be divisible by this value.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.

    Returns:
        A PatchTransformerND configured for grid-to-grid prediction.
    """
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
    """Create a patch-based classifier for 1D, 2D, or 3D grids.

    The input dense spatial grid is split into non-overlapping patches, each embedded as a token.
    After transformer processing, patch tokens are pooled into a single class prediction vector.
    Sinusoidal positional encoding is applied to the patch tokens.

    Args:
        input_dim: Number of spatial dimensions (1, 2, or 3).
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        patch_size: Patch size per spatial dimension. A single int is broadcast to all dimensions.
            The spatial dimensions of the input must be divisible by this value.
        embedding_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.

    Returns:
        A PatchTransformerND configured for patch-based classification.
    """
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
