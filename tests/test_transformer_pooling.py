"""Tests for ml_suite.models.transformer.pooling."""

import pytest
import torch

from ml_suite.models.transformer.pooling import TokenPooling


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_token_pooling_default_mode():
    """Default pooling mode should be 'mean'."""
    pool = TokenPooling()
    assert pool.mode == "mean"


def test_token_pooling_rejects_invalid_mode():
    """Unsupported pooling mode must be rejected."""
    with pytest.raises(ValueError, match="Unsupported pooling mode"):
        TokenPooling(mode="median")  # type: ignore[arg-type]


def test_token_pooling_attention_requires_embedding_dim():
    """Attention pooling without embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim is required"):
        TokenPooling(mode="attention")


def test_token_pooling_attention_with_embedding_dim():
    """Attention pooling with embedding_dim should succeed."""
    pool = TokenPooling(mode="attention", embedding_dim=16)
    assert pool.attention_score is not None


def test_token_pooling_non_attention_has_no_attention_score():
    """Non-attention modes should not attach an attention_score module."""
    pool = TokenPooling(mode="mean")
    assert pool.attention_score is None


# ---------------------------------------------------------------------------
# mean pooling
# ---------------------------------------------------------------------------


def test_mean_pooling_output_shape():
    """Mean pooling should produce (batch, embedding_dim)."""
    pool = TokenPooling(mode="mean")
    x = torch.randn(2, 6, 16)
    out = pool(x)
    assert out.shape == (2, 16)


def test_mean_pooling_correct_average():
    """Mean pooling without mask should equal torch.mean over tokens."""
    pool = TokenPooling(mode="mean")
    x = torch.randn(2, 4, 8)
    out = pool(x)
    expected = x.mean(dim=1)
    assert torch.allclose(out, expected)


def test_mean_pooling_with_mask_ignores_padding():
    """Masked mean pooling should ignore padding tokens."""
    pool = TokenPooling(mode="mean")
    x = torch.ones(1, 4, 4)
    mask = torch.tensor([[True, True, False, False]])
    out = pool(x, mask=mask)
    # Two valid tokens of all-ones -> mean should still be all-ones
    assert torch.allclose(out, torch.ones(1, 4))


def test_mean_pooling_with_mask_output_shape():
    """Masked mean pooling should return (batch, embedding_dim)."""
    pool = TokenPooling(mode="mean")
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    out = pool(x, mask=mask)
    assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# max pooling
# ---------------------------------------------------------------------------


def test_max_pooling_output_shape():
    """Max pooling should produce (batch, embedding_dim)."""
    pool = TokenPooling(mode="max")
    x = torch.randn(2, 6, 16)
    out = pool(x)
    assert out.shape == (2, 16)


def test_max_pooling_correct_values():
    """Max pooling without mask should equal torch.max over tokens."""
    pool = TokenPooling(mode="max")
    x = torch.randn(2, 4, 8)
    out = pool(x)
    expected = x.max(dim=1).values
    assert torch.allclose(out, expected)


def test_max_pooling_with_mask_output_shape():
    """Masked max pooling should return (batch, embedding_dim)."""
    pool = TokenPooling(mode="max")
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 3:] = False
    out = pool(x, mask=mask)
    assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# cls pooling
# ---------------------------------------------------------------------------


def test_cls_pooling_output_shape():
    """CLS pooling should produce (batch, embedding_dim)."""
    pool = TokenPooling(mode="cls")
    x = torch.randn(2, 6, 16)
    out = pool(x)
    assert out.shape == (2, 16)


def test_cls_pooling_returns_first_token():
    """CLS pooling should return exactly the first token."""
    pool = TokenPooling(mode="cls")
    x = torch.randn(2, 6, 8)
    out = pool(x)
    assert torch.allclose(out, x[:, 0])


# ---------------------------------------------------------------------------
# last pooling
# ---------------------------------------------------------------------------


def test_last_pooling_output_shape_no_mask():
    """Last-token pooling without a mask should produce (batch, embedding_dim)."""
    pool = TokenPooling(mode="last")
    x = torch.randn(2, 6, 16)
    out = pool(x)
    assert out.shape == (2, 16)


def test_last_pooling_returns_last_token_no_mask():
    """Last-token pooling should return exactly the last token."""
    pool = TokenPooling(mode="last")
    x = torch.randn(2, 6, 8)
    out = pool(x)
    assert torch.allclose(out, x[:, -1])


def test_last_pooling_with_mask_returns_last_valid_token():
    """Last-token pooling with a mask should return the last valid token per item."""
    pool = TokenPooling(mode="last")
    x = torch.arange(12, dtype=torch.float).reshape(1, 6, 2)
    mask = torch.tensor([[True, True, True, False, False, False]])
    out = pool(x, mask=mask)
    # Last valid index is 2; token 2 == [4, 5]
    assert torch.allclose(out, x[:, 2, :])


# ---------------------------------------------------------------------------
# invalid inputs
# ---------------------------------------------------------------------------


def test_token_pooling_rejects_wrong_rank():
    """Non-3D token tensor must be rejected."""
    pool = TokenPooling(mode="mean")
    x = torch.randn(2, 16)
    with pytest.raises(ValueError, match="must have shape"):
        pool(x)


def test_token_pooling_rejects_non_boolean_mask():
    """Float mask must be rejected."""
    pool = TokenPooling(mode="mean")
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6)  # float, not bool
    with pytest.raises(ValueError, match="boolean tensor"):
        pool(x, mask=mask)
