"""Tests for ml_suite.models.transformer.models and presets."""

import pytest
import torch

from ml_suite.models.transformer.models import (
    ConditionedTokenTransformer,
    PatchTransformerND,
    TokenToClassTransformer,
    TokenToTokenTransformer,
    TokenToVectorTransformer,
)
from ml_suite.models.transformer.presets import (
    make_conditioned_point_to_point_model,
    make_patch_classifier,
    make_patch_grid_model,
    make_point_cloud_classifier,
    make_point_to_point_model,
    make_sequence_classifier,
)


# ---------------------------------------------------------------------------
# TokenToTokenTransformer
# ---------------------------------------------------------------------------


def test_token_to_token_output_shape():
    """Output shape must be (batch, tokens, output_dim)."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    out = model(x)
    assert out.shape == (2, 8, 4)


def test_token_to_token_output_shape_with_mask():
    """Output shape must be unchanged when a valid-token mask is provided."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 6:] = False
    out = model(x, mask=mask)
    assert out.shape == (2, 8, 4)


@pytest.mark.parametrize("pe", ["none", "sinusoidal"])
def test_token_to_token_positional_encoding_variants(pe):
    """Token-to-token transformer should work with 'none' and 'sinusoidal' encodings."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding=pe,
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


def test_token_to_token_learned_positional_encoding():
    """Learned positional encoding requires max_length."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="learned",
        max_length=32,
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


def test_token_to_token_rope_positional_encoding():
    """RoPE positional encoding should preserve output shape."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="rope",
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


def test_token_to_token_causal():
    """Causal token-to-token transformer should preserve output shape."""
    model = TokenToTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        causal=True,
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


# ---------------------------------------------------------------------------
# TokenToVectorTransformer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling", ["mean", "max", "cls", "last"])
def test_token_to_vector_output_shape(pooling):
    """Token-to-vector transformer should produce (batch, output_dim)."""
    model = TokenToVectorTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        pooling=pooling,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 4)


def test_token_to_vector_with_mask():
    """Token-to-vector transformer with mask should produce correct output shape."""
    model = TokenToVectorTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 5:] = False
    assert model(x, mask=mask).shape == (2, 4)


# ---------------------------------------------------------------------------
# TokenToClassTransformer
# ---------------------------------------------------------------------------


def test_token_to_class_output_shape():
    """Token-to-class transformer should produce (batch, num_classes)."""
    model = TokenToClassTransformer(
        input_dim=6,
        num_classes=10,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 10)


def test_token_to_class_rejects_non_positive_num_classes():
    """Non-positive num_classes must be rejected."""
    with pytest.raises(ValueError, match="num_classes must be positive"):
        TokenToClassTransformer(
            input_dim=6,
            num_classes=0,
            embedding_dim=16,
            depth=2,
            num_heads=4,
        )


def test_token_to_class_is_subclass_of_vector():
    """TokenToClassTransformer must be a subclass of TokenToVectorTransformer."""
    model = TokenToClassTransformer(
        input_dim=6,
        num_classes=5,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    assert isinstance(model, TokenToVectorTransformer)


# ---------------------------------------------------------------------------
# ConditionedTokenTransformer
# ---------------------------------------------------------------------------


def test_conditioned_token_no_conditioning_output_shape():
    """With no conditioning, output shape should be (batch, tokens, output_dim)."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


def test_conditioned_token_time_conditioning_output_shape():
    """Time-conditioned model should produce (batch, tokens, output_dim)."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        time_conditioning=True,
    )
    x = torch.randn(2, 8, 6)
    time = torch.rand(2)
    assert model(x, time=time).shape == (2, 8, 4)


def test_conditioned_token_class_conditioning_output_shape():
    """Class-conditioned model should produce (batch, tokens, output_dim)."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        num_classes=10,
    )
    x = torch.randn(2, 8, 6)
    labels = torch.randint(0, 10, (2,))
    assert model(x, class_labels=labels).shape == (2, 8, 4)


def test_conditioned_token_global_context_output_shape():
    """Global-context-conditioned model should produce (batch, tokens, output_dim)."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        global_context_dim=8,
    )
    x = torch.randn(2, 8, 6)
    ctx = torch.randn(2, 8)
    assert model(x, global_context=ctx).shape == (2, 8, 4)


def test_conditioned_token_cross_attention_output_shape():
    """Cross-attention model should attend to a separate context tensor."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 6)
    context = torch.randn(2, 10, 32)
    assert model(x, cross_context=context).shape == (2, 8, 4)


def test_conditioned_token_rejects_missing_cross_context():
    """Missing cross_context when cross_attention_dim is set must raise."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 6)
    with pytest.raises(ValueError, match="cross_context is required"):
        model(x)


def test_conditioned_token_rejects_unexpected_cross_context():
    """Providing cross_context when cross_attention_dim is None must raise."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
    )
    x = torch.randn(2, 8, 6)
    context = torch.randn(2, 10, 32)
    with pytest.raises(ValueError, match="cross_attention_dim is None"):
        model(x, cross_context=context)


def test_conditioned_token_all_conditioning_combined():
    """All conditioning sources together should produce the correct output shape."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="none",
        time_conditioning=True,
        num_classes=10,
        global_context_dim=8,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 8, 6)
    out = model(
        x,
        time=torch.rand(2),
        class_labels=torch.randint(0, 10, (2,)),
        global_context=torch.randn(2, 8),
        cross_context=torch.randn(2, 10, 32),
    )
    assert out.shape == (2, 8, 4)


def test_conditioned_token_nope_alias():
    """'nope' should be accepted as an alias for 'none' positional encoding."""
    model = ConditionedTokenTransformer(
        input_dim=6,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        positional_encoding="nope",
    )
    x = torch.randn(2, 8, 6)
    assert model(x).shape == (2, 8, 4)


# ---------------------------------------------------------------------------
# PatchTransformerND
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dim, x_shape, patch_size, expected_output_shape",
    [
        (1, (2, 3, 32), 4, (2, 3, 32)),
        (2, (2, 3, 16, 16), 4, (2, 3, 16, 16)),
        (3, (2, 3, 8, 8, 8), 2, (2, 3, 8, 8, 8)),
    ],
)
def test_patch_transformer_grid_mode_output_shape(
    input_dim, x_shape, patch_size, expected_output_shape
):
    """Grid-mode PatchTransformerND should reconstruct the original spatial shape."""
    model = PatchTransformerND(
        input_dim=input_dim,
        in_channels=3,
        out_channels=3,
        patch_size=patch_size,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        output_mode="grid",
    )
    x = torch.randn(*x_shape)
    out = model(x)
    assert out.shape == expected_output_shape


def test_patch_transformer_vector_mode_output_shape():
    """Vector-mode PatchTransformerND should produce (batch, vector_output_dim)."""
    model = PatchTransformerND(
        input_dim=2,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        output_mode="vector",
        vector_output_dim=10,
    )
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 10)


def test_patch_transformer_rejects_invalid_output_mode():
    """Unsupported output_mode must be rejected."""
    with pytest.raises(ValueError, match="Unsupported output_mode"):
        PatchTransformerND(
            input_dim=2,
            in_channels=3,
            out_channels=3,
            patch_size=4,
            embedding_dim=16,
            depth=2,
            num_heads=4,
            output_mode="dense",  # type: ignore[arg-type]
        )


def test_patch_transformer_vector_mode_requires_output_dim():
    """vector_output_dim is required when output_mode='vector'."""
    with pytest.raises(ValueError, match="vector_output_dim is required"):
        PatchTransformerND(
            input_dim=2,
            in_channels=3,
            out_channels=3,
            patch_size=4,
            embedding_dim=16,
            depth=2,
            num_heads=4,
            output_mode="vector",
        )


def test_patch_transformer_grid_mode_has_patch_decoder():
    """Grid-mode model should attach a PatchDecoderND."""
    model = PatchTransformerND(
        input_dim=2,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        output_mode="grid",
    )
    assert model.patch_decoder is not None
    assert model.vector_decoder is None


def test_patch_transformer_vector_mode_has_vector_decoder():
    """Vector-mode model should attach a PooledHead."""
    model = PatchTransformerND(
        input_dim=2,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        output_mode="vector",
        vector_output_dim=10,
    )
    assert model.vector_decoder is not None
    assert model.patch_decoder is None


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_make_point_cloud_classifier_output_shape():
    """Point-cloud classifier should produce (batch, num_classes)."""
    model = make_point_cloud_classifier(
        point_dim=3, num_classes=10, embedding_dim=16, depth=2, num_heads=4
    )
    x = torch.randn(2, 64, 3)
    assert model(x).shape == (2, 10)


def test_make_point_cloud_classifier_with_features():
    """Point-cloud classifier with feature_dim should concatenate point coords and features."""
    model = make_point_cloud_classifier(
        point_dim=3, num_classes=10, embedding_dim=16, depth=2, num_heads=4, feature_dim=4
    )
    x = torch.randn(2, 64, 7)  # 3 + 4
    assert model(x).shape == (2, 10)


def test_make_point_cloud_classifier_has_no_positional_encoding():
    """Point-cloud classifier should use no positional encoding (permutation-invariant)."""
    from torch import nn

    model = make_point_cloud_classifier(
        point_dim=3, num_classes=10, embedding_dim=16, depth=2, num_heads=4
    )
    # "none" encoding means absolute_position is nn.Identity and stack uses no rope
    assert isinstance(model.absolute_position, nn.Identity)
    assert model.stack.blocks[0].self_attention.positional_encoding == "none"


def test_make_point_to_point_model_output_shape():
    """Point-to-point model should produce (batch, points, output_dim)."""
    model = make_point_to_point_model(
        point_dim=3, output_dim=4, embedding_dim=16, depth=2, num_heads=4
    )
    x = torch.randn(2, 64, 3)
    assert model(x).shape == (2, 64, 4)


def test_make_point_to_point_model_has_no_positional_encoding():
    """Point-to-point model should use no positional encoding (permutation-equivariant)."""
    from torch import nn

    model = make_point_to_point_model(
        point_dim=3, output_dim=4, embedding_dim=16, depth=2, num_heads=4
    )
    assert isinstance(model.absolute_position, nn.Identity)
    assert model.stack.blocks[0].self_attention.positional_encoding == "none"


def test_make_conditioned_point_to_point_model_no_conditioning():
    """Conditioned point-to-point model without any conditioning should work normally."""
    model = make_conditioned_point_to_point_model(
        point_dim=3, output_dim=4, embedding_dim=16, depth=2, num_heads=4
    )
    x = torch.randn(2, 64, 3)
    assert model(x).shape == (2, 64, 4)


def test_make_conditioned_point_to_point_model_with_time():
    """Conditioned point-to-point model with time conditioning should produce correct shape."""
    model = make_conditioned_point_to_point_model(
        point_dim=3,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        time_conditioning=True,
    )
    x = torch.randn(2, 64, 3)
    assert model(x, time=torch.rand(2)).shape == (2, 64, 4)


def test_make_conditioned_point_to_point_model_with_cross_attention():
    """Conditioned point-to-point model with cross attention should produce correct shape."""
    model = make_conditioned_point_to_point_model(
        point_dim=3,
        output_dim=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 64, 3)
    context = torch.randn(2, 10, 32)
    assert model(x, cross_context=context).shape == (2, 64, 4)


def test_make_sequence_classifier_output_shape():
    """Sequence classifier should produce (batch, num_classes)."""
    model = make_sequence_classifier(
        input_dim=6,
        num_classes=10,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        max_length=64,
    )
    x = torch.randn(2, 16, 6)
    assert model(x).shape == (2, 10)


def test_make_sequence_classifier_uses_rope():
    """Sequence classifier should use RoPE positional encoding inside each block."""
    model = make_sequence_classifier(
        input_dim=6,
        num_classes=10,
        embedding_dim=16,
        depth=2,
        num_heads=4,
        max_length=64,
    )
    assert model.stack.blocks[0].self_attention.positional_encoding == "rope"


def test_make_patch_grid_model_output_shape():
    """Patch grid model should reconstruct the input spatial shape."""
    model = make_patch_grid_model(
        input_dim=2,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 16, 16)


def test_make_patch_classifier_output_shape():
    """Patch classifier should produce (batch, num_classes)."""
    model = make_patch_classifier(
        input_dim=2,
        in_channels=3,
        num_classes=10,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 10)


def test_make_patch_grid_model_1d():
    """Patch grid model should work for 1D inputs."""
    model = make_patch_grid_model(
        input_dim=1,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        embedding_dim=16,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(2, 4, 32)
    out = model(x)
    assert out.shape == (2, 4, 32)
