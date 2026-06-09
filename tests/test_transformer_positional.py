"""Tests for ml_suite.models.transformer.positional."""

import pytest
import torch
from torch import nn

from ml_suite.models.transformer.positional import (
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEmbedding,
    apply_rope_to_tensor,
    build_absolute_positional_embedding,
)


# ---------------------------------------------------------------------------
# LearnedPositionalEmbedding
# ---------------------------------------------------------------------------


def test_learned_pe_output_shape():
    """Output shape must match input shape."""
    pe = LearnedPositionalEmbedding(max_length=16, embedding_dim=8)
    x = torch.randn(2, 6, 8)
    out = pe(x)
    assert out.shape == (2, 6, 8)


def test_learned_pe_adds_embedding():
    """Output should differ from input (positional bias is added)."""
    torch.manual_seed(0)
    pe = LearnedPositionalEmbedding(max_length=16, embedding_dim=8)
    x = torch.randn(2, 6, 8)
    out = pe(x)
    assert not torch.allclose(out, x)


def test_learned_pe_rejects_non_positive_max_length():
    """Non-positive max_length must be rejected."""
    with pytest.raises(ValueError, match="max_length must be positive"):
        LearnedPositionalEmbedding(max_length=0, embedding_dim=8)


def test_learned_pe_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        LearnedPositionalEmbedding(max_length=16, embedding_dim=0)


def test_learned_pe_rejects_sequence_exceeding_max_length():
    """Sequence longer than max_length must be rejected at forward time."""
    pe = LearnedPositionalEmbedding(max_length=4, embedding_dim=8)
    x = torch.randn(2, 8, 8)
    with pytest.raises(ValueError, match="Sequence length"):
        pe(x)


def test_learned_pe_rejects_wrong_rank_input():
    """Non-3D input must be rejected at forward time."""
    pe = LearnedPositionalEmbedding(max_length=16, embedding_dim=8)
    x = torch.randn(2, 8)
    with pytest.raises(ValueError, match="must have shape"):
        pe(x)


def test_learned_pe_at_max_length_boundary():
    """A sequence exactly at max_length should pass."""
    pe = LearnedPositionalEmbedding(max_length=6, embedding_dim=8)
    x = torch.randn(2, 6, 8)
    out = pe(x)
    assert out.shape == (2, 6, 8)


# ---------------------------------------------------------------------------
# SinusoidalPositionalEmbedding
# ---------------------------------------------------------------------------


def test_sinusoidal_pe_output_shape():
    """Output shape must match input shape."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=16)
    x = torch.randn(2, 8, 16)
    out = pe(x)
    assert out.shape == (2, 8, 16)


def test_sinusoidal_pe_adds_encoding():
    """Output should differ from input."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=16)
    x = torch.zeros(2, 8, 16)
    out = pe(x)
    assert not torch.all(out == 0)


def test_sinusoidal_pe_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        SinusoidalPositionalEmbedding(embedding_dim=0)


def test_sinusoidal_pe_rejects_non_positive_max_length():
    """Non-positive max_length must be rejected at construction."""
    with pytest.raises(ValueError, match="max_length must be positive"):
        SinusoidalPositionalEmbedding(embedding_dim=8, max_length=0)


def test_sinusoidal_pe_unbounded_when_no_max_length():
    """Without max_length, arbitrarily long sequences should be accepted."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=8)
    x = torch.randn(1, 1000, 8)
    out = pe(x)
    assert out.shape == (1, 1000, 8)


def test_sinusoidal_pe_rejects_sequence_exceeding_max_length():
    """Sequence longer than max_length must be rejected."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=8, max_length=4)
    x = torch.randn(2, 8, 8)
    with pytest.raises(ValueError, match="Sequence length"):
        pe(x)


def test_sinusoidal_pe_odd_embedding_dim():
    """Odd embedding dimensions should be handled via zero-padding."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=9)
    x = torch.randn(2, 4, 9)
    out = pe(x)
    assert out.shape == (2, 4, 9)


def test_sinusoidal_pe_rejects_wrong_rank_input():
    """Non-3D input must be rejected at forward time."""
    pe = SinusoidalPositionalEmbedding(embedding_dim=8)
    x = torch.randn(2, 8)
    with pytest.raises(ValueError, match="must have shape"):
        pe(x)


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------


def test_rotary_embedding_output_shapes():
    """cos and sin should have shape (seq_len, head_dim // 2)."""
    rope = RotaryEmbedding(head_dim=8)
    cos, sin = rope(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (6, 4)
    assert sin.shape == (6, 4)


def test_rotary_embedding_rejects_non_positive_head_dim():
    """Non-positive head_dim must be rejected."""
    with pytest.raises(ValueError, match="head_dim must be positive"):
        RotaryEmbedding(head_dim=0)


def test_rotary_embedding_rejects_non_positive_base():
    """Non-positive base must be rejected."""
    with pytest.raises(ValueError, match="base must be positive"):
        RotaryEmbedding(head_dim=8, base=0.0)


def test_rotary_embedding_odd_head_dim():
    """Odd head_dim should be handled by using head_dim - 1 rotary dimensions."""
    rope = RotaryEmbedding(head_dim=9)
    cos, sin = rope(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape[0] == 4


# ---------------------------------------------------------------------------
# apply_rope_to_tensor
# ---------------------------------------------------------------------------


def test_apply_rope_preserves_shape():
    """RoPE application must not change the tensor shape."""
    rope = RotaryEmbedding(head_dim=8)
    cos, sin = rope(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 4, 6, 8)
    out = apply_rope_to_tensor(x, cos, sin)
    assert out.shape == (2, 4, 6, 8)


def test_apply_rope_rejects_non_4d_tensor():
    """Non-4D tensor must be rejected."""
    rope = RotaryEmbedding(head_dim=8)
    cos, sin = rope(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 6, 8)
    with pytest.raises(ValueError, match="must have shape"):
        apply_rope_to_tensor(x, cos, sin)


# ---------------------------------------------------------------------------
# build_absolute_positional_embedding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "nope"])
def test_build_pe_none_returns_identity(mode):
    """'none' and 'nope' modes should return nn.Identity."""
    pe = build_absolute_positional_embedding(mode, embedding_dim=8, max_length=None)
    assert isinstance(pe, nn.Identity)


def test_build_pe_rope_returns_identity():
    """'rope' mode is handled inside attention; factory should return nn.Identity."""
    pe = build_absolute_positional_embedding("rope", embedding_dim=8, max_length=None)
    assert isinstance(pe, nn.Identity)


def test_build_pe_learned_returns_learned_embedding():
    """'learned' mode should return a LearnedPositionalEmbedding."""
    pe = build_absolute_positional_embedding("learned", embedding_dim=8, max_length=32)
    assert isinstance(pe, LearnedPositionalEmbedding)


def test_build_pe_learned_requires_max_length():
    """'learned' mode must reject None max_length."""
    with pytest.raises(ValueError, match="max_length is required"):
        build_absolute_positional_embedding("learned", embedding_dim=8, max_length=None)


def test_build_pe_sinusoidal_returns_sinusoidal_embedding():
    """'sinusoidal' mode should return a SinusoidalPositionalEmbedding."""
    pe = build_absolute_positional_embedding("sinusoidal", embedding_dim=8, max_length=None)
    assert isinstance(pe, SinusoidalPositionalEmbedding)


def test_build_pe_sinusoidal_with_max_length():
    """'sinusoidal' with a max_length should pass it to SinusoidalPositionalEmbedding."""
    pe = build_absolute_positional_embedding("sinusoidal", embedding_dim=8, max_length=64)
    assert isinstance(pe, SinusoidalPositionalEmbedding)
    assert pe.max_length == 64
