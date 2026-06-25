"""Reusable encoder and decoder stages for plain and conditioned U-Nets."""

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ml_suite.models.convolution import ConvBlock, ConditionedConvBlock

from .attention import SpatialAttentionBlock
from .types import NormType, ShapePolicy, SkipMode, UpsampleMode
from .utils import get_conv_transpose_nd, make_pointwise_conv, match_spatial_shape


class UNetStage(nn.Module):
    """A stack of ConvBlock or ConditionedConvBlock modules plus optional attention."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_dim: int,
        num_blocks: int,
        block_cls: type[nn.Module] = ConvBlock,
        condition_dim: int | None = None,
        first_stride: int = 1,
        norm_type: NormType = "batch",
        activation: str = "silu",
        num_groups: int = 32,
        attention: SpatialAttentionBlock | None = None,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("UNetStage requires at least one block.")
        if issubclass(block_cls, ConditionedConvBlock) and condition_dim is None:
            raise ValueError("condition_dim is required for ConditionedConvBlock stages.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_dim = conv_dim
        self.num_blocks = num_blocks
        self.block_cls = block_cls
        self.condition_dim = condition_dim
        self.first_stride = first_stride
        self.is_conditioned = issubclass(block_cls, ConditionedConvBlock)

        blocks: list[nn.Module] = []
        blocks.append(
            self._make_block(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=first_stride,
                do_residual=False,
                conv_dim=conv_dim,
                activation=activation,
                norm_type=norm_type,
                num_groups=num_groups,
            )
        )
        for _ in range(num_blocks - 1):
            blocks.append(
                self._make_block(
                    input_channels=output_channels,
                    output_channels=output_channels,
                    stride=1,
                    do_residual=True,
                    conv_dim=conv_dim,
                    activation=activation,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.attention = attention

    def _make_block(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        do_residual: bool,
        conv_dim: int,
        activation: str,
        norm_type: NormType,
        num_groups: int,
    ) -> nn.Module:
        kwargs = dict(
            input_channels=input_channels,
            output_channels=output_channels,
            conv_dim=conv_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=activation,
            norm_type=norm_type,
            num_groups=num_groups,
            do_residual=do_residual,
        )
        if self.is_conditioned:
            assert self.condition_dim is not None
            return self.block_cls(context_dim=self.condition_dim, **kwargs)
        return self.block_cls(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply all conv blocks then the optional spatial attention block.

        Args:
            x: Feature map of shape (batch, input_channels, *spatial).
            condition: FiLM vector of shape (batch, condition_dim). Required for conditioned stages.
            cross_context: Cross-attention token sequence of shape
                (batch, tokens, cross_attention_dim).
            cross_context_mask: Boolean mask of shape (batch, tokens). True = valid token.

        Returns:
            Feature map of shape (batch, output_channels, *spatial).
        """
        out = x
        for block in self.blocks:
            if isinstance(block, ConditionedConvBlock):
                if condition is None:
                    raise ValueError("condition is required for conditioned U-Net stages.")
                out = block(out, condition)
            else:
                out = block(out)

        if self.attention is not None:
            out = self.attention(
                out,
                cross_context=cross_context,
                cross_context_mask=cross_context_mask,
            )
        return out

    def __repr__(self) -> str:
        return (
            f"UNetStage(input_channels={self.input_channels}, "
            f"output_channels={self.output_channels}, "
            f"conv_dim={self.conv_dim}, "
            f"num_blocks={self.num_blocks}, "
            f"conditioned={self.is_conditioned}, "
            f"first_stride={self.first_stride}, "
            f"attention={self.attention is not None})"
        )


class UNetUpStage(nn.Module):
    """One decoder stage shared by plain and conditioned U-Nets."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        conv_dim: int,
        num_blocks: int,
        block_cls: type[nn.Module] = ConvBlock,
        condition_dim: int | None = None,
        upsample_mode: UpsampleMode = "interpolate",
        skip_mode: SkipMode = "concat",
        shape_policy: ShapePolicy = "resize",
        norm_type: NormType = "batch",
        activation: str = "silu",
        num_groups: int = 32,
        attention: SpatialAttentionBlock | None = None,
    ) -> None:
        super().__init__()
        if upsample_mode not in ("interpolate", "transpose"):
            raise ValueError(
                f"upsample_mode must be 'interpolate' or 'transpose'. Got {upsample_mode}."
            )
        if skip_mode not in ("concat", "add"):
            raise ValueError(f"skip_mode must be 'concat' or 'add'. Got {skip_mode}.")
        if shape_policy not in ("resize", "error"):
            raise ValueError(f"shape_policy must be 'resize' or 'error'. Got {shape_policy}.")

        self.input_channels = input_channels
        self.skip_channels = skip_channels
        self.output_channels = output_channels
        self.conv_dim = conv_dim
        self.upsample_mode = upsample_mode
        self.skip_mode = skip_mode
        self.shape_policy = shape_policy

        if upsample_mode == "transpose":
            ConvTranspose = get_conv_transpose_nd(conv_dim)
            self.upsample = ConvTranspose(
                in_channels=input_channels,
                out_channels=skip_channels,
                kernel_size=2,
                stride=2,
            )
            upsampled_channels = skip_channels
            self.channel_projection = nn.Identity()
        else:
            self.upsample = None
            upsampled_channels = input_channels
            if skip_mode == "add" and input_channels != skip_channels:
                self.channel_projection = make_pointwise_conv(
                    conv_dim, input_channels, skip_channels
                )
                upsampled_channels = skip_channels
            else:
                self.channel_projection = nn.Identity()

        if skip_mode == "concat":
            merged_channels = upsampled_channels + skip_channels
        else:
            if upsampled_channels != skip_channels:
                raise ValueError(
                    "skip_mode='add' requires upsampled channels to equal skip_channels. "
                    f"Got upsampled_channels={upsampled_channels}, skip_channels={skip_channels}."
                )
            merged_channels = skip_channels

        self.stage = UNetStage(
            input_channels=merged_channels,
            output_channels=output_channels,
            conv_dim=conv_dim,
            num_blocks=num_blocks,
            block_cls=block_cls,
            condition_dim=condition_dim,
            first_stride=1,
            norm_type=norm_type,
            activation=activation,
            num_groups=num_groups,
            attention=attention,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor | None = None,
        cross_context: torch.Tensor | None = None,
        cross_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upsample x, merge with the skip connection, and apply the decoder stage.

        Args:
            x: Lower-resolution feature map of shape (batch, input_channels, *spatial).
            skip: Matching encoder skip tensor of shape (batch, skip_channels, *spatial_skip).
            condition: FiLM vector of shape (batch, condition_dim). Required for conditioned stages.
            cross_context: Cross-attention token sequence of shape
                (batch, tokens, cross_attention_dim).
            cross_context_mask: Boolean mask of shape (batch, tokens). True = valid token.

        Returns:
            Feature map of shape (batch, output_channels, *spatial_skip).
        """
        if self.upsample_mode == "transpose":
            out = self.upsample(x)
        else:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
            out = self.channel_projection(out)

        out = match_spatial_shape(out, skip, self.shape_policy)

        if self.skip_mode == "concat":
            out = torch.cat([out, skip], dim=1)
        else:
            if out.shape != skip.shape:
                raise ValueError(
                    f"Additive skip requires identical shapes. Got decoder {out.shape} "
                    f"and skip {skip.shape}."
                )
            out = out + skip

        return self.stage(
            out,
            condition=condition,
            cross_context=cross_context,
            cross_context_mask=cross_context_mask,
        )

    def __repr__(self) -> str:
        return (
            f"UNetUpStage(input_channels={self.input_channels}, "
            f"skip_channels={self.skip_channels}, "
            f"output_channels={self.output_channels}, "
            f"conv_dim={self.conv_dim}, "
            f"upsample_mode='{self.upsample_mode}', "
            f"skip_mode='{self.skip_mode}', "
            f"shape_policy='{self.shape_policy}')"
        )
