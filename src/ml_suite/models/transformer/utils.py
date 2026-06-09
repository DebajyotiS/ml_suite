"""Utility functions for token-centric transformer modules."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch

from .types import PositionalEncodingMode


def normalize_positional_encoding_mode(
    mode: PositionalEncodingMode,
) -> PositionalEncodingMode:
    """Normalize aliases for positional encoding modes."""
    if mode == "nope":
        return "none"
    return mode


def validate_token_tensor(x: torch.Tensor, name: str = "x") -> None:
    """Validate a token tensor with shape (batch, tokens, dim)."""
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape (batch, tokens, dim). Got {x.shape}.")


def validate_mask(mask: torch.Tensor, x: torch.Tensor, name: str = "mask") -> None:
    """Validate a boolean valid-token mask for a token tensor."""
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, tokens). Got {mask.shape}.")

    if mask.shape != x.shape[:2]:
        raise ValueError(
            f"{name} must have shape {x.shape[:2]} to match token tensor. Got {mask.shape}."
        )

    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor.")


def valid_mask_to_key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert True-valid masks to PyTorch True-ignore key padding masks."""
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor.")
    return ~mask


def ensure_tuple(value: int | Sequence[int], ndim: int, name: str) -> tuple[int, ...]:
    """Convert an int or sequence into an ndim-length tuple."""
    if isinstance(value, int):
        result = (value,) * ndim
    elif not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an int or sequence of ints. Got {type(value)}.")
    else:
        result = tuple(value)

    if len(result) != ndim:
        raise ValueError(f"{name} must have length {ndim}. Got {result}.")

    if any(v <= 0 for v in result):
        raise ValueError(f"{name} entries must be positive. Got {result}.")

    return result


def compute_patch_grid(
    spatial_shape: Sequence[int],
    patch_size: Sequence[int],
) -> tuple[int, ...]:
    """Compute patch grid shape for a spatial tensor."""
    spatial = tuple(spatial_shape)
    patch = tuple(patch_size)

    if len(spatial) != len(patch):
        raise ValueError(
            f"spatial_shape and patch_size must have the same length. Got {spatial} and {patch}."
        )

    if any(s % p != 0 for s, p in zip(spatial, patch, strict=True)):
        raise ValueError(f"spatial_shape {spatial} must be divisible by patch_size {patch}.")

    return tuple(s // p for s, p in zip(spatial, patch, strict=True))


def num_tokens_from_grid(grid_shape: Sequence[int]) -> int:
    """Return the flattened number of tokens from a patch grid shape."""
    return prod(grid_shape)


def last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last valid token for each batch item."""
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor.")

    flipped = mask.long() * torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return flipped.argmax(dim=1)
