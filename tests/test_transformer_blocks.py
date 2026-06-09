"""Tests for ml_suite.models.transformer.blocks and stacks."""

import pytest
import torch
from torch import nn

from ml_suite.models.transformer.blocks import (
    FeedForward,
    TransformerBlock,
    build_norm,
)
from ml_suite.models.transformer.stacks import TransformerStack


# ---------------------------------------------------------------------------
# build_norm
# ---------------------------------------------------------------------------


def test_build_norm_layer_returns_layer_norm():
    """'layer' norm type should produce an nn.LayerNorm."""
    norm = build_norm("layer", 16)
    assert isinstance(norm, nn.LayerNorm)


def test_build_norm_rms_returns_rms_norm():
    """'rms' norm type should produce an nn.RMSNorm."""
    norm = build_norm("rms", 16)
    assert isinstance(norm, nn.RMSNorm)


def test_build_norm_invalid_type_raises():
    """Unsupported norm type must be rejected."""
    with pytest.raises(ValueError, match="Unsupported norm_type"):
        build_norm("batch", 16)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


def test_feedforward_output_shape():
    """Output shape must match input shape for a rank-3 tensor."""
    ff = FeedForward(embedding_dim=16, hidden_dim=64)
    x = torch.randn(2, 8, 16)
    out = ff(x)
    assert out.shape == (2, 8, 16)


def test_feedforward_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        FeedForward(embedding_dim=0, hidden_dim=64)


def test_feedforward_rejects_non_positive_hidden_dim():
    """Non-positive hidden_dim must be rejected."""
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        FeedForward(embedding_dim=16, hidden_dim=0)


def test_feedforward_rejects_invalid_dropout():
    """Dropout >= 1.0 must be rejected."""
    with pytest.raises(ValueError, match="dropout must be in"):
        FeedForward(embedding_dim=16, hidden_dim=64, dropout=1.0)


# ---------------------------------------------------------------------------
# TransformerBlock — self-attention only
# ---------------------------------------------------------------------------


def test_transformer_block_output_shape_no_cross_attention():
    """Output shape must match input shape when cross-attention is disabled."""
    block = TransformerBlock(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 8, 16)


def test_transformer_block_output_shape_with_mask():
    """Output shape must be unchanged when a padding mask is provided."""
    block = TransformerBlock(embedding_dim=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 6:] = False
    out = block(x, mask=mask)
    assert out.shape == (2, 8, 16)


def test_transformer_block_causal_output_shape():
    """Causal block should preserve shape."""
    block = TransformerBlock(embedding_dim=16, num_heads=4, causal=True)
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 8, 16)


def test_transformer_block_rms_norm_output_shape():
    """RMSNorm variant should preserve shape."""
    block = TransformerBlock(embedding_dim=16, num_heads=4, norm_type="rms")
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 8, 16)


def test_transformer_block_rejects_context_when_cross_attention_disabled():
    """Providing context to a self-attention-only block must raise."""
    block = TransformerBlock(embedding_dim=16, num_heads=4, use_cross_attention=False)
    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="use_cross_attention.*False"):
        block(x, context=context)


def test_transformer_block_rejects_missing_cross_attention_dim():
    """use_cross_attention=True without cross_attention_dim must be rejected."""
    with pytest.raises(ValueError, match="cross_attention_dim is required"):
        TransformerBlock(embedding_dim=16, num_heads=4, use_cross_attention=True)


def test_transformer_block_rejects_non_positive_cross_attention_dim():
    """Non-positive cross_attention_dim must be rejected."""
    with pytest.raises(ValueError, match="cross_attention_dim must be positive"):
        TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            use_cross_attention=True,
            cross_attention_dim=0,
        )


def test_transformer_block_rejects_non_positive_mlp_ratio():
    """Non-positive mlp_ratio must be rejected."""
    with pytest.raises(ValueError, match="mlp_ratio must be positive"):
        TransformerBlock(embedding_dim=16, num_heads=4, mlp_ratio=-1.0)


# ---------------------------------------------------------------------------
# TransformerBlock — cross-attention
# ---------------------------------------------------------------------------


def test_transformer_block_cross_attention_output_shape():
    """Cross-attended block should preserve query tensor shape."""
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        use_cross_attention=True,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 10, 32)
    out = block(x, context=context)
    assert out.shape == (2, 8, 16)


def test_transformer_block_cross_attention_with_context_mask():
    """Cross-attention with context mask should preserve shape."""
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        use_cross_attention=True,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 10, 32)
    context_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask[0, 8:] = False
    out = block(x, context=context, context_mask=context_mask)
    assert out.shape == (2, 8, 16)


def test_transformer_block_cross_attention_requires_context_at_forward():
    """Omitting context when use_cross_attention=True must raise."""
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        use_cross_attention=True,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="context is required"):
        block(x)


def test_transformer_block_use_cross_attention_attribute():
    """use_cross_attention property should reflect constructor argument."""
    block = TransformerBlock(
        embedding_dim=16,
        num_heads=4,
        use_cross_attention=True,
        cross_attention_dim=32,
    )
    assert block.use_cross_attention is True


# ---------------------------------------------------------------------------
# TransformerStack
# ---------------------------------------------------------------------------


def test_transformer_stack_output_shape():
    """Stack output shape must match input shape."""
    stack = TransformerStack(embedding_dim=16, depth=3, num_heads=4)
    x = torch.randn(2, 8, 16)
    out = stack(x)
    assert out.shape == (2, 8, 16)


def test_transformer_stack_depth_matches_block_count():
    """Number of transformer blocks must equal depth."""
    stack = TransformerStack(embedding_dim=16, depth=5, num_heads=4)
    assert stack.depth == 5
    assert len(stack.blocks) == 5


def test_transformer_stack_final_norm_applied():
    """With final_norm=True the final_norm module should not be Identity."""
    stack = TransformerStack(embedding_dim=16, depth=2, num_heads=4, final_norm=True)
    assert not isinstance(stack.final_norm, nn.Identity)


def test_transformer_stack_no_final_norm_is_identity():
    """With final_norm=False the final_norm module should be Identity."""
    stack = TransformerStack(embedding_dim=16, depth=2, num_heads=4, final_norm=False)
    assert isinstance(stack.final_norm, nn.Identity)


def test_transformer_stack_rejects_depth_less_than_one():
    """depth < 1 must be rejected."""
    with pytest.raises(ValueError, match="depth must be at least 1"):
        TransformerStack(embedding_dim=16, depth=0, num_heads=4)


def test_transformer_stack_with_mask():
    """Stack with a padding mask should produce the correct output shape."""
    stack = TransformerStack(embedding_dim=16, depth=2, num_heads=4)
    x = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 6:] = False
    out = stack(x, mask=mask)
    assert out.shape == (2, 8, 16)


def test_transformer_stack_cross_attention_output_shape():
    """Stack with cross-attention should produce correct output shape."""
    stack = TransformerStack(
        embedding_dim=16,
        depth=2,
        num_heads=4,
        use_cross_attention=True,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 10, 32)
    out = stack(x, context=context)
    assert out.shape == (2, 8, 16)


def test_transformer_stack_rejects_cross_attention_without_dim():
    """use_cross_attention=True without cross_attention_dim must be rejected."""
    with pytest.raises(ValueError, match="cross_attention_dim is required"):
        TransformerStack(
            embedding_dim=16,
            depth=2,
            num_heads=4,
            use_cross_attention=True,
        )


@pytest.mark.parametrize("norm_type", ["layer", "rms"])
def test_transformer_stack_norm_types(norm_type):
    """Stack should accept both 'layer' and 'rms' norm types."""
    stack = TransformerStack(embedding_dim=16, depth=2, num_heads=4, norm_type=norm_type)
    x = torch.randn(2, 8, 16)
    out = stack(x)
    assert out.shape == (2, 8, 16)
