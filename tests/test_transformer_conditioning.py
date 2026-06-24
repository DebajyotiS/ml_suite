"""Tests for ml_suite.models.transformer.conditioning."""

import pytest
import torch

from ml_suite.models.transformer.conditioning import (
    ConditionTokenProjector,
    TransformerConditioningBuilder,
)


# ---------------------------------------------------------------------------
# TransformerConditioningBuilder : construction and has_conditioning
# ---------------------------------------------------------------------------


def test_conditioning_builder_no_conditioning_has_conditioning_false():
    """Builder with no conditioning sources should report has_conditioning=False."""
    builder = TransformerConditioningBuilder(embedding_dim=16)
    assert not builder.has_conditioning()


def test_conditioning_builder_time_has_conditioning_true():
    """Builder with time conditioning should report has_conditioning=True."""
    builder = TransformerConditioningBuilder(embedding_dim=16, time_conditioning=True)
    assert builder.has_conditioning()


def test_conditioning_builder_classes_has_conditioning_true():
    """Builder with class conditioning should report has_conditioning=True."""
    builder = TransformerConditioningBuilder(embedding_dim=16, num_classes=10)
    assert builder.has_conditioning()


def test_conditioning_builder_global_context_has_conditioning_true():
    """Builder with global context should report has_conditioning=True."""
    builder = TransformerConditioningBuilder(embedding_dim=16, global_context_dim=8)
    assert builder.has_conditioning()


def test_conditioning_builder_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        TransformerConditioningBuilder(embedding_dim=0)


def test_conditioning_builder_rejects_non_positive_num_classes():
    """Non-positive num_classes must be rejected."""
    with pytest.raises(ValueError, match="num_classes must be positive"):
        TransformerConditioningBuilder(embedding_dim=16, num_classes=0)


def test_conditioning_builder_rejects_class_dropout_without_num_classes():
    """class_dropout_prob > 0 without num_classes must be rejected."""
    with pytest.raises(ValueError, match="class_dropout_prob requires num_classes"):
        TransformerConditioningBuilder(embedding_dim=16, class_dropout_prob=0.1)


def test_conditioning_builder_rejects_invalid_class_dropout():
    """class_dropout_prob >= 1.0 must be rejected."""
    with pytest.raises(ValueError, match="class_dropout_prob must be in"):
        TransformerConditioningBuilder(embedding_dim=16, num_classes=10, class_dropout_prob=1.0)


def test_conditioning_builder_rejects_non_positive_global_context_dim():
    """Non-positive global_context_dim must be rejected."""
    with pytest.raises(ValueError, match="global_context_dim must be positive"):
        TransformerConditioningBuilder(embedding_dim=16, global_context_dim=0)


# ---------------------------------------------------------------------------
# TransformerConditioningBuilder : forward pass
# ---------------------------------------------------------------------------


def test_conditioning_builder_no_conditioning_returns_zeros():
    """With no conditioning, output should be a zero tensor."""
    builder = TransformerConditioningBuilder(embedding_dim=16)
    out = builder(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2, 16)
    assert torch.all(out == 0)


def test_conditioning_builder_time_conditioning_output_shape():
    """Time-conditioned builder should return (batch, embedding_dim)."""
    builder = TransformerConditioningBuilder(embedding_dim=16, time_conditioning=True)
    out = builder(batch_size=2, device=torch.device("cpu"), dtype=torch.float32, time=torch.rand(2))
    assert out.shape == (2, 16)


def test_conditioning_builder_class_conditioning_output_shape():
    """Class-conditioned builder should return (batch, embedding_dim)."""
    builder = TransformerConditioningBuilder(embedding_dim=16, num_classes=10)
    out = builder(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        class_labels=torch.randint(0, 10, (2,)),
    )
    assert out.shape == (2, 16)


def test_conditioning_builder_global_context_output_shape():
    """Global-context-conditioned builder should return (batch, embedding_dim)."""
    builder = TransformerConditioningBuilder(embedding_dim=16, global_context_dim=8)
    out = builder(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_context=torch.randn(2, 8),
    )
    assert out.shape == (2, 16)


def test_conditioning_builder_all_conditioning_combined():
    """All three conditioning sources should be summed into one (batch, embedding_dim) tensor."""
    builder = TransformerConditioningBuilder(
        embedding_dim=16,
        time_conditioning=True,
        num_classes=10,
        global_context_dim=8,
    )
    out = builder(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        time=torch.rand(2),
        class_labels=torch.randint(0, 10, (2,)),
        global_context=torch.randn(2, 8),
    )
    assert out.shape == (2, 16)


def test_conditioning_builder_rejects_time_without_time_conditioning():
    """Providing time when time_conditioning=False must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16, time_conditioning=False)
    with pytest.raises(ValueError, match="time_conditioning=False"):
        builder(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time=torch.rand(2),
        )


def test_conditioning_builder_rejects_missing_time():
    """Missing time when time_conditioning=True must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16, time_conditioning=True)
    with pytest.raises(ValueError, match="time must be provided"):
        builder(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)


def test_conditioning_builder_rejects_class_labels_without_num_classes():
    """Providing class_labels when num_classes is None must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16)
    with pytest.raises(ValueError, match="num_classes is None"):
        builder(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            class_labels=torch.zeros(2, dtype=torch.long),
        )


def test_conditioning_builder_rejects_missing_class_labels():
    """Missing class_labels when num_classes is set must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16, num_classes=10)
    with pytest.raises(ValueError, match="class_labels must be provided"):
        builder(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)


def test_conditioning_builder_rejects_global_context_without_dim():
    """Providing global_context when global_context_dim is None must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16)
    with pytest.raises(ValueError, match="global_context_dim is None"):
        builder(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            global_context=torch.randn(2, 8),
        )


def test_conditioning_builder_rejects_missing_global_context():
    """Missing global_context when global_context_dim is set must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16, global_context_dim=8)
    with pytest.raises(ValueError, match="global_context must be provided"):
        builder(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)


def test_conditioning_builder_rejects_time_batch_mismatch():
    """time batch size mismatching the requested batch_size must raise."""
    builder = TransformerConditioningBuilder(embedding_dim=16, time_conditioning=True)
    with pytest.raises(ValueError, match="time batch size"):
        builder(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time=torch.rand(3),
        )


# ---------------------------------------------------------------------------
# ConditionTokenProjector
# ---------------------------------------------------------------------------


def test_condition_token_projector_output_shape_single_token():
    """Default single-token projection should produce (batch, 1, embedding_dim)."""
    proj = ConditionTokenProjector(context_dim=8, embedding_dim=16)
    out = proj(torch.randn(2, 8))
    assert out.shape == (2, 1, 16)


def test_condition_token_projector_output_shape_multi_token():
    """Multi-token projection should produce (batch, num_tokens, embedding_dim)."""
    proj = ConditionTokenProjector(context_dim=8, embedding_dim=16, num_tokens=4)
    out = proj(torch.randn(2, 8))
    assert out.shape == (2, 4, 16)


def test_condition_token_projector_rejects_non_positive_context_dim():
    """Non-positive context_dim must be rejected."""
    with pytest.raises(ValueError, match="context_dim must be positive"):
        ConditionTokenProjector(context_dim=0, embedding_dim=16)


def test_condition_token_projector_rejects_non_positive_embedding_dim():
    """Non-positive embedding_dim must be rejected."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        ConditionTokenProjector(context_dim=8, embedding_dim=0)


def test_condition_token_projector_rejects_non_positive_num_tokens():
    """Non-positive num_tokens must be rejected."""
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        ConditionTokenProjector(context_dim=8, embedding_dim=16, num_tokens=0)


def test_condition_token_projector_rejects_wrong_rank():
    """Non-2D context must be rejected."""
    proj = ConditionTokenProjector(context_dim=8, embedding_dim=16)
    with pytest.raises(ValueError, match="must have shape"):
        proj(torch.randn(2, 4, 8))


def test_condition_token_projector_rejects_wrong_context_dim():
    """Context with incorrect feature dimension must be rejected."""
    proj = ConditionTokenProjector(context_dim=8, embedding_dim=16)
    with pytest.raises(ValueError, match="must have shape"):
        proj(torch.randn(2, 12))
