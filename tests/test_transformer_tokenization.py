"""Tests for ml_suite.models.transformer.tokenization."""

import pytest
import torch

from ml_suite.models.transformer.tokenization import (
    ContinuousInputTokenizer,
    DiscreteTokenTokenizer,
    PatchTokenizerND,
    SetTokenizer,
)


# ---------------------------------------------------------------------------
# ContinuousInputTokenizer
# ---------------------------------------------------------------------------


def test_continuous_tokenizer_output_shape():
    """Output should be (batch, tokens, embedding_dim)."""
    tok = ContinuousInputTokenizer(input_dim=6, embedding_dim=16)
    x = torch.randn(2, 8, 6)
    out = tok(x)
    assert out.shape == (2, 8, 16)


def test_continuous_tokenizer_stores_dims():
    """input_dim and embedding_dim should be stored as attributes."""
    tok = ContinuousInputTokenizer(input_dim=6, embedding_dim=16)
    assert tok.input_dim == 6
    assert tok.embedding_dim == 16


def test_continuous_tokenizer_multi_layer_output_shape():
    """Multi-layer MLP tokenizer should still produce the correct output shape."""
    tok = ContinuousInputTokenizer(input_dim=6, embedding_dim=16, num_layers=3, hidden_dim=32)
    x = torch.randn(2, 8, 6)
    out = tok(x)
    assert out.shape == (2, 8, 16)


def test_continuous_tokenizer_rejects_wrong_rank():
    """Non-3D input must be rejected at forward time."""
    tok = ContinuousInputTokenizer(input_dim=6, embedding_dim=16)
    x = torch.randn(2, 6)
    with pytest.raises(ValueError, match="must have shape"):
        tok(x)


def test_continuous_tokenizer_rejects_wrong_input_dim():
    """Mismatched last dimension must be rejected at forward time."""
    tok = ContinuousInputTokenizer(input_dim=6, embedding_dim=16)
    x = torch.randn(2, 8, 8)
    with pytest.raises(ValueError, match="input_dim=6"):
        tok(x)


# ---------------------------------------------------------------------------
# SetTokenizer
# ---------------------------------------------------------------------------


def test_set_tokenizer_is_subclass_of_continuous():
    """SetTokenizer should inherit from ContinuousInputTokenizer."""
    tok = SetTokenizer(input_dim=4, embedding_dim=8)
    assert isinstance(tok, ContinuousInputTokenizer)


def test_set_tokenizer_output_shape():
    """SetTokenizer should produce the same output shape as ContinuousInputTokenizer."""
    tok = SetTokenizer(input_dim=4, embedding_dim=8)
    x = torch.randn(2, 10, 4)
    out = tok(x)
    assert out.shape == (2, 10, 8)


# ---------------------------------------------------------------------------
# DiscreteTokenTokenizer
# ---------------------------------------------------------------------------


def test_discrete_tokenizer_output_shape():
    """Output should be (batch, tokens, embedding_dim)."""
    tok = DiscreteTokenTokenizer(vocab_size=100, embedding_dim=16)
    ids = torch.randint(0, 100, (2, 8))
    out = tok(ids)
    assert out.shape == (2, 8, 16)


def test_discrete_tokenizer_stores_dims():
    """vocab_size and embedding_dim should be stored as attributes."""
    tok = DiscreteTokenTokenizer(vocab_size=100, embedding_dim=16)
    assert tok.vocab_size == 100
    assert tok.embedding_dim == 16


def test_discrete_tokenizer_rejects_non_positive_vocab_size():
    """Non-positive vocab_size must be rejected."""
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        DiscreteTokenTokenizer(vocab_size=0, embedding_dim=16)


def test_discrete_tokenizer_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        DiscreteTokenTokenizer(vocab_size=100, embedding_dim=0)


def test_discrete_tokenizer_rejects_wrong_rank():
    """Non-2D token IDs must be rejected at forward time."""
    tok = DiscreteTokenTokenizer(vocab_size=100, embedding_dim=16)
    ids = torch.randint(0, 100, (2, 8, 1))
    with pytest.raises(ValueError, match="must have shape"):
        tok(ids)


def test_discrete_tokenizer_rejects_out_of_range_ids():
    """Token IDs outside [0, vocab_size) must be rejected."""
    tok = DiscreteTokenTokenizer(vocab_size=10, embedding_dim=16)
    ids = torch.tensor([[0, 5, 10]])  # 10 is out of range for vocab_size=10
    with pytest.raises(ValueError, match="must be in"):
        tok(ids)


def test_discrete_tokenizer_rejects_negative_ids():
    """Negative token IDs must be rejected."""
    tok = DiscreteTokenTokenizer(vocab_size=10, embedding_dim=16)
    ids = torch.tensor([[0, -1, 5]])
    with pytest.raises(ValueError, match="must be in"):
        tok(ids)


def test_discrete_tokenizer_with_padding_idx():
    """Padding idx should be passed to the embedding without error."""
    tok = DiscreteTokenTokenizer(vocab_size=10, embedding_dim=8, padding_idx=0)
    ids = torch.randint(0, 10, (2, 4))
    out = tok(ids)
    assert out.shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# PatchTokenizerND
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spatial_dim, input_shape, patch_size, expected_tokens, expected_grid",
    [
        (1, (2, 3, 32), 4, 8, (8,)),
        (2, (2, 3, 16, 16), 4, 16, (4, 4)),
        (3, (2, 3, 8, 8, 8), 2, 64, (4, 4, 4)),
    ],
)
def test_patch_tokenizer_output_shape(
    spatial_dim, input_shape, patch_size, expected_tokens, expected_grid
):
    """PatchTokenizerND should return correct token tensor and grid shape."""
    tok = PatchTokenizerND(
        spatial_dim=spatial_dim,
        in_channels=3,
        embedding_dim=16,
        patch_size=patch_size,
    )
    x = torch.randn(*input_shape)
    tokens, grid_shape = tok(x)

    assert tokens.shape == (input_shape[0], expected_tokens, 16)
    assert grid_shape == expected_grid


def test_patch_tokenizer_anisotropic_patch_size_2d():
    """Anisotropic patch sizes should be supported for 2D inputs."""
    tok = PatchTokenizerND(
        spatial_dim=2,
        in_channels=3,
        embedding_dim=16,
        patch_size=(4, 8),
    )
    x = torch.randn(2, 3, 16, 32)
    tokens, grid_shape = tok(x)
    assert tokens.shape == (2, 4 * 4, 16)
    assert grid_shape == (4, 4)


def test_patch_tokenizer_rejects_invalid_spatial_dim():
    """spatial_dim outside {1, 2, 3} must be rejected."""
    with pytest.raises(ValueError, match="spatial_dim must be"):
        PatchTokenizerND(spatial_dim=4, in_channels=3, embedding_dim=16, patch_size=4)


def test_patch_tokenizer_rejects_non_positive_in_channels():
    """Non-positive in_channels must be rejected."""
    with pytest.raises(ValueError, match="in_channels must be positive"):
        PatchTokenizerND(spatial_dim=2, in_channels=0, embedding_dim=16, patch_size=4)


def test_patch_tokenizer_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        PatchTokenizerND(spatial_dim=2, in_channels=3, embedding_dim=0, patch_size=4)


def test_patch_tokenizer_rejects_wrong_rank_input():
    """Input with incorrect number of dimensions must be rejected."""
    tok = PatchTokenizerND(spatial_dim=2, in_channels=3, embedding_dim=16, patch_size=4)
    x = torch.randn(2, 3, 16)  # rank 3 instead of 4
    with pytest.raises(ValueError, match="Expected 4D input"):
        tok(x)


def test_patch_tokenizer_rejects_wrong_channel_count():
    """Mismatched channel dimension must be rejected."""
    tok = PatchTokenizerND(spatial_dim=2, in_channels=3, embedding_dim=16, patch_size=4)
    x = torch.randn(2, 6, 16, 16)  # 6 channels instead of 3
    with pytest.raises(ValueError, match="Expected 3 channels"):
        tok(x)
