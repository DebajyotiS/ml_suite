"""Shared BaseUNet implementation used by plain and conditioned U-Nets."""

from collections.abc import Sequence

import torch
from torch import nn

from ml_suite.models.convolution import ConvBlock, ConditionedConvBlock
from ml_suite.utils.activations import get_activation

from .attention import SpatialAttentionBlock
from .stages import UNetStage, UNetUpStage
from .types import (
    AttentionLocation,
    AttentionType,
    DownsampleMode,
    NormType,
    ShapePolicy,
    SkipMode,
    UpsampleMode,
)
from .utils import (
    make_pointwise_conv,
    make_pool,
    resolve_blocks_per_stage,
    validate_conv_dim,
    validate_spatial_input,
)


class BaseUNet(nn.Module):
    """Shared encoder-decoder mechanics for UNet and ConditionedUNet."""

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        stage_channels: Sequence[int],
        blocks_per_stage: int | Sequence[int] = 2,
        block_cls: type[nn.Module] = ConvBlock,
        condition_dim: int | None = None,
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
        attention_type: AttentionType = "self",
        num_attention_heads: int = 4,
        attention_head_dim: int | None = None,
        cross_attention_dim: int | None = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            conv_dim=conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            stage_channels=stage_channels,
            block_cls=block_cls,
            condition_dim=condition_dim,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
            skip_mode=skip_mode,
            shape_policy=shape_policy,
            attention_downsample_factors=attention_downsample_factors,
            attention_locations=attention_locations,
            attention_type=attention_type,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            attention_dropout=attention_dropout,
        )

        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_channels = list(stage_channels)
        self.block_cls = block_cls
        self.condition_dim = condition_dim
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.skip_mode = skip_mode
        self.shape_policy = shape_policy
        self.norm_type = norm_type
        self.activation = activation
        self.num_groups = num_groups
        self.output_activation_name = output_activation
        self.attention_downsample_factors = set(attention_downsample_factors)
        self.attention_locations = attention_locations
        self.attention_type = attention_type
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.attention_dropout = attention_dropout
        self.is_conditioned = issubclass(block_cls, ConditionedConvBlock)
        self.uses_cross_attention = attention_type in ("cross", "self_cross") and bool(
            attention_downsample_factors
        )

        self.blocks_per_stage = resolve_blocks_per_stage(
            blocks_per_stage,
            len(self.stage_channels),
        )
        self.encoder_pools = self._build_encoder_pools()
        self.encoder_stages = self._build_encoder_stages()
        self.decoder_stages = self._build_decoder_stages()
        self.output_projection = make_pointwise_conv(
            conv_dim=conv_dim,
            input_channels=self.stage_channels[0],
            output_channels=out_channels,
        )
        self.output_activation = (
            nn.Identity() if output_activation is None else get_activation(output_activation)
        )

    @staticmethod
    def _validate_init_args(
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        stage_channels: Sequence[int],
        block_cls: type[nn.Module],
        condition_dim: int | None,
        downsample_mode: str,
        upsample_mode: str,
        skip_mode: str,
        shape_policy: str,
        attention_downsample_factors: Sequence[int],
        attention_locations: str,
        attention_type: str,
        num_attention_heads: int,
        attention_head_dim: int | None,
        cross_attention_dim: int | None,
        attention_dropout: float,
    ) -> None:
        validate_conv_dim(conv_dim)
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive. Got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive. Got {out_channels}.")
        if len(stage_channels) < 2:
            raise ValueError("UNet requires at least two stage channels.")
        if any(ch <= 0 for ch in stage_channels):
            raise ValueError(f"All stage channels must be positive. Got {stage_channels}.")
        if issubclass(block_cls, ConditionedConvBlock) and condition_dim is None:
            raise ValueError("condition_dim is required for conditioned U-Net blocks.")
        if condition_dim is not None and condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive. Got {condition_dim}.")
        if downsample_mode not in ("stride", "pool"):
            raise ValueError(f"downsample_mode must be 'stride' or 'pool'. Got {downsample_mode}.")
        if upsample_mode not in ("interpolate", "transpose"):
            raise ValueError(
                f"upsample_mode must be 'interpolate' or 'transpose'. Got {upsample_mode}."
            )
        if skip_mode not in ("concat", "add"):
            raise ValueError(f"skip_mode must be 'concat' or 'add'. Got {skip_mode}.")
        if shape_policy not in ("resize", "error"):
            raise ValueError(f"shape_policy must be 'resize' or 'error'. Got {shape_policy}.")
        if any(factor < 1 for factor in attention_downsample_factors):
            raise ValueError("attention_downsample_factors must contain positive integers.")
        if attention_locations not in ("encoder", "decoder", "both", "bottleneck"):
            raise ValueError(
                "attention_locations must be one of 'encoder', 'decoder', 'both', "
                f"or 'bottleneck'. Got {attention_locations}."
            )
        if attention_type not in ("self", "cross", "self_cross"):
            raise ValueError(
                f"attention_type must be 'self', 'cross', or 'self_cross'. Got {attention_type}."
            )
        if attention_type in ("cross", "self_cross") and cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim is required when attention_type is 'cross' or 'self_cross'."
            )
        if attention_type == "self" and cross_attention_dim is not None:
            raise ValueError("cross_attention_dim was provided, but attention_type='self'.")
        if cross_attention_dim is not None and cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive. Got {cross_attention_dim}.")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive. Got {num_attention_heads}.")
        if attention_head_dim is not None and attention_head_dim <= 0:
            raise ValueError(f"attention_head_dim must be positive. Got {attention_head_dim}.")
        if attention_dropout < 0.0 or attention_dropout >= 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1). Got {attention_dropout}.")
        if attention_type in ("cross", "self_cross") and len(attention_downsample_factors) == 0:
            raise ValueError("Cross-attention requires at least one attention downsample factor.")

    def _build_encoder_pools(self) -> nn.ModuleList:
        pools = nn.ModuleList()
        if self.downsample_mode == "pool":
            for _ in range(len(self.stage_channels) - 1):
                pools.append(make_pool(self.conv_dim))
        return pools

    def _should_use_attention(
        self,
        downsample_factor: int,
        location: str,
    ) -> bool:
        if downsample_factor not in self.attention_downsample_factors:
            return False
        if location == "bottleneck":
            return self.attention_locations in ("both", "bottleneck")
        if location == "encoder":
            return self.attention_locations in ("both", "encoder")
        if location == "decoder":
            return self.attention_locations in ("both", "decoder")
        raise ValueError(f"Unsupported attention location: {location}")

    def _make_attention(
        self,
        channels: int,
        downsample_factor: int,
        location: str,
    ) -> nn.Module | None:
        if not self._should_use_attention(downsample_factor, location):
            return None
        return SpatialAttentionBlock(
            channels=channels,
            attention_type=self.attention_type,
            num_heads=self.num_attention_heads,
            head_dim=self.attention_head_dim,
            cross_attention_dim=self.cross_attention_dim,
            dropout=self.attention_dropout,
        )

    def _build_encoder_stages(self) -> nn.ModuleList:
        stages = nn.ModuleList()
        current_channels = self.in_channels
        num_stages = len(self.stage_channels)
        for stage_idx, target_channels in enumerate(self.stage_channels):
            downsample_factor = 2**stage_idx
            is_bottleneck = stage_idx == num_stages - 1
            first_stride = 2 if stage_idx > 0 and self.downsample_mode == "stride" else 1
            attention_location = "bottleneck" if is_bottleneck else "encoder"
            attention = self._make_attention(target_channels, downsample_factor, attention_location)
            stages.append(
                UNetStage(
                    input_channels=current_channels,
                    output_channels=target_channels,
                    conv_dim=self.conv_dim,
                    num_blocks=self.blocks_per_stage[stage_idx],
                    block_cls=self.block_cls,
                    condition_dim=self.condition_dim,
                    first_stride=first_stride,
                    norm_type=self.norm_type,
                    activation=self.activation,
                    num_groups=self.num_groups,
                    attention=attention,
                )
            )
            current_channels = target_channels
        return stages

    def _build_decoder_stages(self) -> nn.ModuleList:
        stages = nn.ModuleList()
        current_channels = self.stage_channels[-1]
        for skip_stage_idx in range(len(self.stage_channels) - 2, -1, -1):
            skip_channels = self.stage_channels[skip_stage_idx]
            downsample_factor = 2**skip_stage_idx
            attention = self._make_attention(skip_channels, downsample_factor, "decoder")
            stages.append(
                UNetUpStage(
                    input_channels=current_channels,
                    skip_channels=skip_channels,
                    output_channels=skip_channels,
                    conv_dim=self.conv_dim,
                    num_blocks=self.blocks_per_stage[skip_stage_idx],
                    block_cls=self.block_cls,
                    condition_dim=self.condition_dim,
                    upsample_mode=self.upsample_mode,
                    skip_mode=self.skip_mode,
                    shape_policy=self.shape_policy,
                    norm_type=self.norm_type,
                    activation=self.activation,
                    num_groups=self.num_groups,
                    attention=attention,
                )
            )
            current_channels = skip_channels
        return stages

    def _validate_cross_inputs(
        self,
        batch_size: int,
        cross_context: torch.Tensor | None,
        cross_context_mask: torch.Tensor | None,
    ) -> None:
        if self.uses_cross_attention and cross_context is None:
            raise ValueError("cross_context must be provided when cross-attention is enabled.")
        if not self.uses_cross_attention and cross_context is not None:
            raise ValueError("cross_context was provided, but cross-attention is not enabled.")
        if cross_context is not None:
            if cross_context.ndim != 3:
                raise ValueError(
                    f"cross_context must have shape (batch, tokens, dim). Got {cross_context.shape}."
                )
            if cross_context.shape[0] != batch_size:
                raise ValueError(
                    f"cross_context batch size ({cross_context.shape[0]}) must match "
                    f"input batch size ({batch_size})."
                )
            if cross_context.shape[2] != self.cross_attention_dim:
                raise ValueError(
                    f"cross_context feature dimension must be {self.cross_attention_dim}. "
                    f"Got {cross_context.shape[2]}."
                )
        if cross_context_mask is not None:
            if cross_context is None:
                raise ValueError("cross_context_mask was provided, but cross_context is None.")
            if cross_context_mask.shape != cross_context.shape[:2]:
                raise ValueError(
                    f"cross_context_mask must have shape {cross_context.shape[:2]}. "
                    f"Got {cross_context_mask.shape}."
                )
            if cross_context_mask.dtype != torch.bool:
                raise ValueError("cross_context_mask must be a boolean tensor.")

    def _forward_impl(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_spatial_input(x, self.conv_dim, self.in_channels)
        if self.is_conditioned and condition is None:
            raise ValueError("condition is required for conditioned U-Net blocks.")
        if not self.is_conditioned and condition is not None:
            raise ValueError("condition was provided for an unconditioned U-Net.")
        self._validate_cross_inputs(x.shape[0], cross_context, cross_context_mask)

        skips: list[torch.Tensor] = []
        out = x
        for stage_idx, stage in enumerate(self.encoder_stages):
            if stage_idx > 0 and self.downsample_mode == "pool":
                out = self.encoder_pools[stage_idx - 1](out)
            out = stage(
                out,
                condition=condition,
                cross_context=cross_context,
                cross_context_mask=cross_context_mask,
            )
            if stage_idx < len(self.encoder_stages) - 1:
                skips.append(out)

        for decoder_stage, skip in zip(self.decoder_stages, reversed(skips), strict=True):
            out = decoder_stage(
                out,
                skip=skip,
                condition=condition,
                cross_context=cross_context,
                cross_context_mask=cross_context_mask,
            )

        return self.output_activation(self.output_projection(out))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(conv_dim={self.conv_dim}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"stage_channels={self.stage_channels}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"downsample_mode='{self.downsample_mode}', "
            f"upsample_mode='{self.upsample_mode}', "
            f"skip_mode='{self.skip_mode}', "
            f"shape_policy='{self.shape_policy}', "
            f"norm_type='{self.norm_type}', "
            f"activation='{self.activation}', "
            f"attention_downsample_factors={sorted(self.attention_downsample_factors)}, "
            f"attention_type='{self.attention_type}')"
        )
