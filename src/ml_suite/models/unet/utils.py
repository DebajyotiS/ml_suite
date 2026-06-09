"""Dimension-agnostic utilities used by U-Net modules."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .types import ShapePolicy


def validate_conv_dim(conv_dim: int) -> None:
    if conv_dim not in (1, 2, 3):
        raise ValueError(f"conv_dim must be 1, 2, or 3. Got {conv_dim}.")


def get_conv_nd(conv_dim: int) -> type[nn.Module]:
    validate_conv_dim(conv_dim)
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[conv_dim]


def get_pool_nd(conv_dim: int) -> type[nn.Module]:
    validate_conv_dim(conv_dim)
    return {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[conv_dim]


def get_conv_transpose_nd(conv_dim: int) -> type[nn.Module]:
    validate_conv_dim(conv_dim)
    return {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[conv_dim]


def make_pointwise_conv(
    conv_dim: int,
    input_channels: int,
    output_channels: int,
) -> nn.Module:
    Conv = get_conv_nd(conv_dim)
    return Conv(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def make_pool(conv_dim: int) -> nn.Module:
    Pool = get_pool_nd(conv_dim)
    return Pool(kernel_size=2, stride=2)


def match_spatial_shape(
    x: torch.Tensor,
    target: torch.Tensor,
    policy: ShapePolicy = "resize",
) -> torch.Tensor:
    x_spatial = x.shape[2:]
    target_spatial = target.shape[2:]

    if x_spatial == target_spatial:
        return x

    if policy == "error":
        raise ValueError(
            f"Spatial shape {x_spatial} does not match target spatial shape {target_spatial}."
        )

    if policy != "resize":
        raise ValueError(f"Unknown shape_policy: {policy}.")

    return F.interpolate(x, size=target_spatial, mode="nearest")


def resolve_blocks_per_stage(
    blocks_per_stage: int | Sequence[int],
    num_stages: int,
) -> list[int]:
    if isinstance(blocks_per_stage, int):
        resolved = [blocks_per_stage] * num_stages
    else:
        if len(blocks_per_stage) != num_stages:
            raise ValueError(
                f"Length of blocks_per_stage ({len(blocks_per_stage)}) must match "
                f"number of stages ({num_stages})."
            )
        resolved = list(blocks_per_stage)

    if any(n < 1 for n in resolved):
        raise ValueError("Every stage must contain at least one block.")

    return resolved


def validate_spatial_input(
    x: torch.Tensor,
    conv_dim: int,
    expected_channels: int,
) -> None:
    if x.ndim != conv_dim + 2:
        raise ValueError(
            f"Expected {conv_dim + 2}D input for conv_dim={conv_dim}, "
            f"but got tensor with shape {x.shape}."
        )

    if x.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} input channels, but got {x.shape[1]}.")
