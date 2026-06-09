"""Tests for ml_suite.models.transformer.utils."""

import pytest
import torch

from ml_suite.models.transformer.utils import (
    compute_patch_grid,
    ensure_tuple,
    last_valid_indices,
    normalize_positional_encoding_mode,
    num_tokens_from_grid,
    valid_mask_to_key_padding_mask,
    validate_mask,
    validate_token_tensor,
)


def test_normalize_mode_nope_becomes_none():
    """'nope' alias should be normalised to 'none'."""
    assert normalize_positional_encoding_mode("nope") == "none"


@pytest.mark.parametrize("mode", ["none", "learned", "sinusoidal", "rope"])
def test_normalize_mode_passthrough(mode):
    """Non-alias modes should be returned unchanged."""
    assert normalize_positional_encoding_mode(mode) == mode


def test_validate_token_tensor_accepts_rank3():
    """A rank-3 tensor (batch, tokens, dim) should pass validation."""
    x = torch.randn(2, 4, 8)
    validate_token_tensor(x)  # should not raise


@pytest.mark.parametrize("shape", [(4, 8), (2, 4, 8, 1)])
def test_validate_token_tensor_rejects_wrong_rank(shape):
    """Rank-2 and rank-4 tensors must be rejected."""
    x = torch.randn(*shape)
    with pytest.raises(ValueError, match="must have shape"):
        validate_token_tensor(x)


def test_validate_token_tensor_custom_name_in_error():
    """Custom name should appear in the error message."""
    x = torch.randn(4, 8)
    with pytest.raises(ValueError, match="my_input"):
        validate_token_tensor(x, name="my_input")


def test_validate_mask_accepts_valid_boolean_mask():
    """A boolean mask shaped (batch, tokens) should pass validation."""
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    validate_mask(mask, x)  # should not raise


def test_validate_mask_rejects_non_boolean():
    """Float masks must be rejected."""
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)  # float
    with pytest.raises(ValueError, match="boolean tensor"):
        validate_mask(mask, x)


def test_validate_mask_rejects_wrong_rank():
    """Rank-3 mask must be rejected."""
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="must have shape"):
        validate_mask(mask, x)


def test_validate_mask_rejects_shape_mismatch():
    """Mask (batch, tokens) shape must match the token tensor prefix."""
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)  # wrong token count
    with pytest.raises(ValueError, match="must have shape"):
        validate_mask(mask, x)


def test_valid_mask_to_key_padding_mask_none_returns_none():
    """None input should pass through as None."""
    assert valid_mask_to_key_padding_mask(None) is None


def test_valid_mask_to_key_padding_mask_inverts_boolean():
    """True-valid mask should become True-ignore (inverted) mask."""
    mask = torch.tensor([[True, True, False, False]])
    result = valid_mask_to_key_padding_mask(mask)
    expected = torch.tensor([[False, False, True, True]])
    assert torch.equal(result, expected)


def test_valid_mask_to_key_padding_mask_rejects_non_boolean():
    """Non-boolean mask must be rejected."""
    mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="boolean tensor"):
        valid_mask_to_key_padding_mask(mask)


def test_ensure_tuple_broadcasts_int():
    """A single int should be broadcast to an ndim-length tuple."""
    result = ensure_tuple(3, ndim=2, name="patch_size")
    assert result == (3, 3)


def test_ensure_tuple_accepts_list():
    """A list of the correct length should be returned as a tuple."""
    result = ensure_tuple([2, 4], ndim=2, name="patch_size")
    assert result == (2, 4)


def test_ensure_tuple_accepts_tuple():
    """A tuple of the correct length should be returned as-is."""
    result = ensure_tuple((1, 2, 3), ndim=3, name="patch_size")
    assert result == (1, 2, 3)


def test_ensure_tuple_rejects_wrong_length():
    """A sequence with incorrect length must be rejected."""
    with pytest.raises(ValueError, match="must have length"):
        ensure_tuple([2, 4, 8], ndim=2, name="patch_size")


def test_ensure_tuple_rejects_non_positive_entries():
    """Zero or negative entries must be rejected."""
    with pytest.raises(ValueError, match="entries must be positive"):
        ensure_tuple([2, 0], ndim=2, name="patch_size")


def test_ensure_tuple_rejects_unsupported_type():
    """Non-int non-sequence types must be rejected."""
    with pytest.raises(TypeError, match="must be an int or sequence"):
        ensure_tuple(3.0, ndim=2, name="patch_size")  # type: ignore[arg-type]


def test_compute_patch_grid_1d():
    """1D patch grid should return a single-element tuple."""
    assert compute_patch_grid((32,), (4,)) == (8,)


def test_compute_patch_grid_2d():
    """2D patch grid should divide each spatial dimension independently."""
    assert compute_patch_grid((32, 64), (4, 8)) == (8, 8)


def test_compute_patch_grid_3d():
    """3D patch grid should work across all three spatial axes."""
    assert compute_patch_grid((8, 16, 32), (2, 4, 8)) == (4, 4, 4)


def test_compute_patch_grid_rejects_length_mismatch():
    """spatial_shape and patch_size must have the same length."""
    with pytest.raises(ValueError, match="same length"):
        compute_patch_grid((32, 32), (4,))


def test_compute_patch_grid_rejects_non_divisible():
    """Spatial dimensions not divisible by patch size must be rejected."""
    with pytest.raises(ValueError, match="divisible"):
        compute_patch_grid((33, 32), (4, 4))


def test_num_tokens_from_grid_2d():
    """Flattened patch count should be the product of grid dimensions."""
    assert num_tokens_from_grid((8, 8)) == 64


def test_num_tokens_from_grid_3d():
    """3D grid should return the product of all three dimensions."""
    assert num_tokens_from_grid((4, 4, 4)) == 64


def test_last_valid_indices_returns_last_true_position():
    """Should return the index of the last True entry per batch item."""
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    indices = last_valid_indices(mask)
    assert indices[0].item() == 2
    assert indices[1].item() == 0


def test_last_valid_indices_rejects_non_boolean():
    """Float mask must be rejected."""
    mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="boolean tensor"):
        last_valid_indices(mask)
