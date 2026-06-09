"""Tests for ml_suite.models.transformer.attention."""

import pytest
import torch

from ml_suite.models.transformer.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention — construction
# ---------------------------------------------------------------------------


def test_self_attention_properties_default_head_dim():
    """head_dim should default to embedding_dim // num_heads."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4)
    assert attn.embedding_dim == 16
    assert attn.num_heads == 4
    assert attn.head_dim == 4
    assert attn.inner_dim == 16


def test_self_attention_properties_custom_head_dim():
    """Custom head_dim should set inner_dim = num_heads * head_dim."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, head_dim=8)
    assert attn.head_dim == 8
    assert attn.inner_dim == 32


def test_self_attention_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        MultiHeadSelfAttention(embedding_dim=0, num_heads=4)


def test_self_attention_rejects_non_positive_num_heads():
    """Non-positive num_heads must be rejected."""
    with pytest.raises(ValueError, match="num_heads must be positive"):
        MultiHeadSelfAttention(embedding_dim=16, num_heads=0)


def test_self_attention_rejects_non_divisible_embedding_dim():
    """embedding_dim not divisible by num_heads must be rejected when head_dim is None."""
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadSelfAttention(embedding_dim=15, num_heads=4)


def test_self_attention_rejects_invalid_dropout():
    """Dropout outside [0, 1) must be rejected."""
    with pytest.raises(ValueError, match="dropout must be in"):
        MultiHeadSelfAttention(embedding_dim=16, num_heads=4, dropout=1.0)


def test_self_attention_rejects_unsupported_positional_encoding():
    """Positional encodings other than 'none' and 'rope' must be rejected."""
    with pytest.raises(ValueError, match="only supports"):
        MultiHeadSelfAttention(embedding_dim=16, num_heads=4, positional_encoding="learned")


def test_self_attention_rope_creates_rope_module():
    """RoPE positional encoding should attach a RotaryEmbedding to the module."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, positional_encoding="rope")
    assert attn.rope is not None


def test_self_attention_none_encoding_has_no_rope():
    """'none' encoding should not attach a RotaryEmbedding."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, positional_encoding="none")
    assert attn.rope is None


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention — forward
# ---------------------------------------------------------------------------


def test_self_attention_output_shape():
    """Output shape must match input shape (batch, tokens, embedding_dim)."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 6, 16)
    out = attn(x)
    assert out.shape == (2, 6, 16)


def test_self_attention_output_shape_with_mask():
    """Output shape must be unchanged when a valid-token mask is provided."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    out = attn(x, mask=mask)
    assert out.shape == (2, 6, 16)


def test_self_attention_causal_output_shape():
    """Causal self-attention should preserve shape."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, causal=True)
    x = torch.randn(2, 6, 16)
    out = attn(x)
    assert out.shape == (2, 6, 16)


def test_self_attention_rope_output_shape():
    """RoPE self-attention should preserve shape."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, positional_encoding="rope")
    x = torch.randn(2, 6, 16)
    out = attn(x)
    assert out.shape == (2, 6, 16)


def test_self_attention_rejects_wrong_rank_input():
    """Non-3D input must be rejected at forward time."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 16)
    with pytest.raises(ValueError, match="must have shape"):
        attn(x)


def test_self_attention_rejects_wrong_embedding_dim():
    """Mismatched embedding dimension must be rejected at forward time."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 6, 32)
    with pytest.raises(ValueError, match="embedding_dim=16"):
        attn(x)


def test_self_attention_causal_with_mask_output_shape():
    """Causal self-attention with a padding mask should combine both masks correctly."""
    attn = MultiHeadSelfAttention(embedding_dim=16, num_heads=4, causal=True)
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 5] = False
    out = attn(x, mask=mask)
    assert out.shape == (2, 6, 16)


@pytest.mark.parametrize(
    "batch,tokens,dim,heads",
    [
        (1, 1, 8, 2),
        (4, 16, 32, 8),
        (2, 8, 16, 4),
    ],
)
def test_self_attention_various_shapes(batch, tokens, dim, heads):
    """Self-attention should handle a variety of batch/token/dim/head configurations."""
    attn = MultiHeadSelfAttention(embedding_dim=dim, num_heads=heads)
    x = torch.randn(batch, tokens, dim)
    out = attn(x)
    assert out.shape == (batch, tokens, dim)


# ---------------------------------------------------------------------------
# MultiHeadCrossAttention — construction
# ---------------------------------------------------------------------------


def test_cross_attention_properties():
    """Properties should reflect the constructor arguments."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    assert attn.query_dim == 16
    assert attn.context_dim == 32
    assert attn.num_heads == 4
    assert attn.head_dim == 4


def test_cross_attention_custom_head_dim():
    """Custom head_dim should override the default division."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4, head_dim=8)
    assert attn.head_dim == 8
    assert attn.inner_dim == 32


def test_cross_attention_rejects_non_positive_query_dim():
    """Non-positive query_dim must be rejected."""
    with pytest.raises(ValueError, match="query_dim must be positive"):
        MultiHeadCrossAttention(query_dim=0, context_dim=32, num_heads=4)


def test_cross_attention_rejects_non_positive_context_dim():
    """Non-positive context_dim must be rejected."""
    with pytest.raises(ValueError, match="context_dim must be positive"):
        MultiHeadCrossAttention(query_dim=16, context_dim=0, num_heads=4)


def test_cross_attention_rejects_non_divisible_query_dim():
    """query_dim not divisible by num_heads must be rejected when head_dim is None."""
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadCrossAttention(query_dim=15, context_dim=32, num_heads=4)


# ---------------------------------------------------------------------------
# MultiHeadCrossAttention — forward
# ---------------------------------------------------------------------------


def test_cross_attention_output_shape():
    """Output shape must be (batch, query_tokens, query_dim)."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 6, 16)
    context = torch.randn(2, 10, 32)
    out = attn(x, context)
    assert out.shape == (2, 6, 16)


def test_cross_attention_output_shape_with_context_mask():
    """Context masking should not alter the output shape."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 6, 16)
    context = torch.randn(2, 10, 32)
    context_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask[0, 7:] = False
    out = attn(x, context, context_mask=context_mask)
    assert out.shape == (2, 6, 16)


def test_cross_attention_rejects_batch_mismatch():
    """Mismatched batch sizes between x and context must be rejected."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 6, 16)
    context = torch.randn(3, 10, 32)
    with pytest.raises(ValueError, match="batch size"):
        attn(x, context)


def test_cross_attention_rejects_wrong_rank_context():
    """2D context tensor must be rejected."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 6, 16)
    context = torch.randn(2, 32)
    with pytest.raises(ValueError, match="must have shape"):
        attn(x, context)


def test_cross_attention_rejects_wrong_context_dim():
    """Context with incorrect last dimension must be rejected."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 6, 16)
    context = torch.randn(2, 10, 64)
    with pytest.raises(ValueError, match="context_dim=32"):
        attn(x, context)


def test_cross_attention_asymmetric_query_context_lengths():
    """Query and context can have different sequence lengths."""
    attn = MultiHeadCrossAttention(query_dim=16, context_dim=32, num_heads=4)
    x = torch.randn(2, 4, 16)
    context = torch.randn(2, 20, 32)
    out = attn(x, context)
    assert out.shape == (2, 4, 16)
