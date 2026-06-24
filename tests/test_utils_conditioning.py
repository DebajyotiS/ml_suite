"""Tests for ml_suite.utils.conditioning (shared time embedding primitives)."""

import pytest
import torch

from ml_suite.utils.conditioning import SinusoidalTimeEmbedding, TimeEmbeddingMLP


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------


def test_sinusoidal_time_embedding_output_shape_1d():
    """1D time tensor (batch,) should produce (batch, embedding_dim)."""
    emb = SinusoidalTimeEmbedding(embedding_dim=16)
    out = emb(torch.rand(4))
    assert out.shape == (4, 16)


def test_sinusoidal_time_embedding_output_shape_2d():
    """2D time tensor (batch, 1) should be squeezed before embedding."""
    emb = SinusoidalTimeEmbedding(embedding_dim=16)
    out = emb(torch.rand(4, 1))
    assert out.shape == (4, 16)


def test_sinusoidal_time_embedding_rejects_wrong_rank():
    """2D time tensor with more than 1 column must be rejected."""
    emb = SinusoidalTimeEmbedding(embedding_dim=16)
    with pytest.raises(ValueError, match="must have shape"):
        emb(torch.rand(4, 2))


def test_sinusoidal_time_embedding_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        SinusoidalTimeEmbedding(embedding_dim=0)


def test_sinusoidal_time_embedding_odd_embedding_dim():
    """Odd embedding dimensions should be handled via zero-padding."""
    emb = SinusoidalTimeEmbedding(embedding_dim=9)
    out = emb(torch.rand(4))
    assert out.shape == (4, 9)


def test_sinusoidal_time_embedding_output_is_float():
    """Output should always be float regardless of input dtype."""
    emb = SinusoidalTimeEmbedding(embedding_dim=16)
    out = emb(torch.arange(4).float())
    assert out.dtype == torch.float32


def test_sinusoidal_time_embedding_frequencies_are_buffered():
    """Frequencies should be registered as a buffer (serialised with state_dict)."""
    emb = SinusoidalTimeEmbedding(embedding_dim=16)
    assert "frequencies" in dict(emb.named_buffers())


# ---------------------------------------------------------------------------
# TimeEmbeddingMLP
# ---------------------------------------------------------------------------


def test_time_embedding_mlp_sinusoidal_output_shape():
    """Sinusoidal TimeEmbeddingMLP should produce (batch, embedding_dim)."""
    mlp = TimeEmbeddingMLP(embedding_dim=16, embedding_type="sinusoidal")
    out = mlp(torch.rand(4))
    assert out.shape == (4, 16)


def test_time_embedding_mlp_learned_output_shape_1d():
    """Learned TimeEmbeddingMLP with (batch,) time should produce (batch, embedding_dim)."""
    mlp = TimeEmbeddingMLP(embedding_dim=16, embedding_type="learned")
    out = mlp(torch.rand(4))
    assert out.shape == (4, 16)


def test_time_embedding_mlp_learned_output_shape_2d():
    """Learned TimeEmbeddingMLP with (batch, 1) time should produce (batch, embedding_dim)."""
    mlp = TimeEmbeddingMLP(embedding_dim=16, embedding_type="learned")
    out = mlp(torch.rand(4, 1))
    assert out.shape == (4, 16)


def test_time_embedding_mlp_learned_rejects_wrong_rank():
    """Learned MLP with 2D time tensor with more than 1 column must be rejected."""
    mlp = TimeEmbeddingMLP(embedding_dim=16, embedding_type="learned")
    with pytest.raises(ValueError, match="must have shape"):
        mlp(torch.rand(4, 2))


def test_time_embedding_mlp_rejects_invalid_embedding_type():
    """Unsupported embedding_type must be rejected."""
    with pytest.raises(ValueError, match="Unsupported embedding_type"):
        TimeEmbeddingMLP(embedding_dim=16, embedding_type="fourier")  # type: ignore[arg-type]


def test_time_embedding_mlp_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        TimeEmbeddingMLP(embedding_dim=0)
