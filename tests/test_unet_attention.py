import pytest
import torch

from ml_suite.models.unet.attention import SpatialAttentionBlock


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 16, 32),
        (2, 16, 16, 16),
        (2, 16, 4, 8, 8),
    ],
)
def test_spatial_self_attention_shapes(input_shape):
    block = SpatialAttentionBlock(channels=16, attention_type="self", num_heads=4)
    x = torch.randn(*input_shape)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("attention_type", ["cross", "self_cross"])
def test_spatial_cross_attention_shapes(attention_type):
    block = SpatialAttentionBlock(
        channels=16,
        attention_type=attention_type,
        num_heads=4,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)

    assert block(x, cross_context=context, cross_context_mask=mask).shape == x.shape


def test_cross_attention_validation():
    block = SpatialAttentionBlock(
        channels=16,
        attention_type="cross",
        num_heads=4,
        cross_attention_dim=32,
    )
    x = torch.randn(2, 16, 8, 8)

    with pytest.raises(ValueError, match="cross_context"):
        block(x)

    with pytest.raises(ValueError, match="shape"):
        block(x, cross_context=torch.randn(2, 32))

    with pytest.raises(ValueError, match="batch"):
        block(x, cross_context=torch.randn(3, 5, 32))

    with pytest.raises(ValueError, match="feature dimension"):
        block(x, cross_context=torch.randn(2, 5, 33))

    with pytest.raises(ValueError, match="mask"):
        block(
            x,
            cross_context=torch.randn(2, 5, 32),
            cross_context_mask=torch.ones(2, 4, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="boolean"):
        block(
            x,
            cross_context=torch.randn(2, 5, 32),
            cross_context_mask=torch.ones(2, 5),
        )


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 16, 32),
        (2, 16, 16, 16),
        (2, 16, 4, 8, 8),
    ],
)
def test_spatial_mask_self_attention_shapes(input_shape):
    block = SpatialAttentionBlock(channels=16, attention_type="self", num_heads=4)
    x = torch.randn(*input_shape)
    spatial_mask = torch.ones(*input_shape[:1], *input_shape[2:], dtype=torch.bool)
    assert block(x, spatial_mask=spatial_mask).shape == x.shape


def test_spatial_mask_with_self_cross():
    block = SpatialAttentionBlock(
        channels=16, attention_type="self_cross", num_heads=4, cross_attention_dim=32
    )
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 5, 32)
    ctx_mask = torch.ones(2, 5, dtype=torch.bool)
    spatial_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    spatial_mask[:, :4, :] = False  # mask top half of patches

    out = block(x, cross_context=context, cross_context_mask=ctx_mask, spatial_mask=spatial_mask)
    assert out.shape == x.shape


def test_spatial_mask_changes_output():
    block = SpatialAttentionBlock(channels=16, attention_type="self", num_heads=4, dropout=0.0)
    block.eval()

    x = torch.randn(1, 16, 8, 8)
    full_mask = torch.ones(1, 8, 8, dtype=torch.bool)
    partial_mask = full_mask.clone()
    partial_mask[:, :4, :] = False

    with torch.no_grad():
        out_full = block(x, spatial_mask=full_mask)
        out_partial = block(x, spatial_mask=partial_mask)

    assert not torch.allclose(out_full, out_partial)


def test_spatial_mask_validation():
    self_block = SpatialAttentionBlock(channels=16, attention_type="self", num_heads=4)
    cross_block = SpatialAttentionBlock(
        channels=16, attention_type="cross", num_heads=4, cross_attention_dim=32
    )
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 5, 32)

    with pytest.raises(ValueError, match="self-attention"):
        cross_block(x, cross_context=context, spatial_mask=torch.ones(2, 8, 8, dtype=torch.bool))

    with pytest.raises(ValueError, match="shape"):
        self_block(x, spatial_mask=torch.ones(2, 4, 4, dtype=torch.bool))

    with pytest.raises(ValueError, match="boolean"):
        self_block(x, spatial_mask=torch.ones(2, 8, 8))


def test_self_attention_rejects_cross_context():
    block = SpatialAttentionBlock(channels=16, attention_type="self", num_heads=4)
    x = torch.randn(2, 16, 8, 8)

    with pytest.raises(ValueError, match="cross_context"):
        block(x, cross_context=torch.randn(2, 5, 32))


def test_attention_constructor_validation():
    with pytest.raises(ValueError, match="cross_attention_dim"):
        SpatialAttentionBlock(16, attention_type="cross", num_heads=4)

    with pytest.raises(ValueError, match="divisible"):
        SpatialAttentionBlock(15, attention_type="self", num_heads=4)

    with pytest.raises(ValueError, match="attention_type"):
        SpatialAttentionBlock(16, attention_type="bad", num_heads=4)  # type: ignore[arg-type]
