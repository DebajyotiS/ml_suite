"""Public U-Net model classes."""

from collections.abc import Sequence

import torch

from ml_suite.models.convolution import ConditionedConvBlock, ConvBlock

from .base import BaseUNet
from .conditioning import ConditioningBuilder
from .types import (
    AttentionLocation,
    AttentionType,
    DownsampleMode,
    NormType,
    ShapePolicy,
    SkipMode,
    TimeEmbeddingType,
    UpsampleMode,
)


class UNet(BaseUNet):
    """A plain dimension-agnostic U-Net for 1D, 2D, or 3D data.

    Supports optional spatial self-attention. Time, class, global context, and
    cross-attention conditioning are handled by ``ConditionedUNet``.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        stage_channels: Sequence[int],
        blocks_per_stage: int | Sequence[int] = 2,
        downsample_mode: DownsampleMode = "stride",
        upsample_mode: UpsampleMode = "interpolate",
        skip_mode: SkipMode = "concat",
        shape_policy: ShapePolicy = "resize",
        norm_type: NormType = "batch",
        activation: str = "silu",
        num_groups: int = 32,
        output_activation: str | None = None,
        attention_downsample_factors: Sequence[int] = (),
        attention_locations: AttentionLocation = "both",
        num_attention_heads: int = 4,
        attention_head_dim: int | None = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            conv_dim=conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            stage_channels=stage_channels,
            blocks_per_stage=blocks_per_stage,
            block_cls=ConvBlock,
            condition_dim=None,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
            skip_mode=skip_mode,
            shape_policy=shape_policy,
            norm_type=norm_type,
            activation=activation,
            num_groups=num_groups,
            output_activation=output_activation,
            attention_downsample_factors=attention_downsample_factors,
            attention_locations=attention_locations,
            attention_type="self",
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=None,
            attention_dropout=attention_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain U-Net encoder-decoder forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, *spatial).

        Returns:
            Output tensor of shape (batch, out_channels, *spatial).
        """
        return self._forward_impl(x)


class ConditionedUNet(BaseUNet):
    """A FiLM-conditioned dimension-agnostic U-Net with optional cross-attention.

    Global conditioning sources ``time``, ``class_labels``, and ``global_context``
    are combined into one vector and passed to every conditioned convolutional
    block. Token conditioning ``cross_context`` is used only by attention blocks.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        stage_channels: Sequence[int],
        blocks_per_stage: int | Sequence[int] = 2,
        downsample_mode: DownsampleMode = "stride",
        upsample_mode: UpsampleMode = "interpolate",
        skip_mode: SkipMode = "concat",
        shape_policy: ShapePolicy = "resize",
        norm_type: NormType = "batch",
        activation: str = "silu",
        num_groups: int = 32,
        output_activation: str | None = None,
        condition_dim: int | None = None,
        time_conditioning: bool = False,
        time_embedding_type: TimeEmbeddingType = "sinusoidal",
        num_classes: int | None = None,
        class_dropout_prob: float = 0.0,
        global_context_dim: int | None = None,
        attention_downsample_factors: Sequence[int] = (),
        attention_locations: AttentionLocation = "both",
        attention_type: AttentionType = "self",
        num_attention_heads: int = 4,
        attention_head_dim: int | None = None,
        cross_attention_dim: int | None = None,
        attention_dropout: float = 0.0,
    ) -> None:
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must contain at least one entry.")

        inferred_condition_dim = (
            condition_dim if condition_dim is not None else 4 * stage_channels[0]
        )

        super().__init__(
            conv_dim=conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            stage_channels=stage_channels,
            blocks_per_stage=blocks_per_stage,
            block_cls=ConditionedConvBlock,
            condition_dim=inferred_condition_dim,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
            skip_mode=skip_mode,
            shape_policy=shape_policy,
            norm_type=norm_type,
            activation=activation,
            num_groups=num_groups,
            output_activation=output_activation,
            attention_downsample_factors=attention_downsample_factors,
            attention_locations=attention_locations,
            attention_type=attention_type,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            attention_dropout=attention_dropout,
        )

        self.conditioning = ConditioningBuilder(
            condition_dim=inferred_condition_dim,
            time_conditioning=time_conditioning,
            time_embedding_type=time_embedding_type,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            global_context_dim=global_context_dim,
        )
        self.condition_dim = inferred_condition_dim
        self.time_conditioning = time_conditioning
        self.time_embedding_type = time_embedding_type
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.global_context_dim = global_context_dim

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the conditioned U-Net forward pass.

        All enabled global conditioning sources are projected to condition_dim and summed into a
        single FiLM vector passed to every conditioned convolutional block. Token cross-context
        is used only by spatial attention blocks.

        Args:
            x: Input tensor of shape (batch, in_channels, *spatial).
            time: Diffusion timestep of shape (batch,). Required when time_conditioning=True.
            class_labels: Integer class indices of shape (batch,). Required when num_classes is set.
            global_context: Global context vector of shape (batch, global_context_dim).
                Required when global_context_dim is set.
            cross_context: Token sequence of shape (batch, tokens, cross_attention_dim).
                Required when attention_type is 'cross' or 'self_cross'.
            cross_context_mask: Boolean valid-token mask of shape (batch, tokens).
                True indicates a valid (non-padded) token.

        Returns:
            Output tensor of shape (batch, out_channels, *spatial).
        """
        condition = self.conditioning(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            time=time,
            class_labels=class_labels,
            global_context=global_context,
        )
        return self._forward_impl(
            x,
            condition=condition,
            cross_context=cross_context,
            cross_context_mask=cross_context_mask,
        )

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            base[:-1]
            + f", condition_dim={self.condition_dim}, "
            + f"time_conditioning={self.time_conditioning}, "
            + f"num_classes={self.num_classes}, "
            + f"global_context_dim={self.global_context_dim})"
        )
