import pytest
import torch

from ml_suite.models.unet import ConditionedUNet
from ml_suite.models.unet.attention import SpatialAttentionBlock


def test_conditioned_unet_time_conditioning_shape():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        condition_dim=32,
        time_conditioning=True,
        attention_downsample_factors=(4,),
        attention_type="self",
        num_attention_heads=4,
    )

    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))

    assert model(x, time=t).shape == x.shape


def test_conditioned_unet_time_shape_b1_works():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        time_conditioning=True,
    )

    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2, 1))

    assert model(x, time=t).shape == x.shape


def test_conditioned_unet_class_conditioning_shape():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        condition_dim=32,
        num_classes=10,
    )

    x = torch.randn(2, 4, 32, 32)
    labels = torch.tensor([1, 3])

    assert model(x, class_labels=labels).shape == x.shape


def test_conditioned_unet_global_context_shape():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=3,
        out_channels=3,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        condition_dim=32,
        global_context_dim=12,
    )

    x = torch.randn(2, 3, 32, 32)
    z = torch.randn(2, 12)

    assert model(x, global_context=z).shape == x.shape


def test_conditioned_unet_cross_attention_shape():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        condition_dim=32,
        time_conditioning=True,
        attention_downsample_factors=(2, 4),
        attention_type="self_cross",
        cross_attention_dim=24,
        num_attention_heads=4,
    )

    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    cross_context = torch.randn(2, 5, 24)
    mask = torch.ones(2, 5, dtype=torch.bool)

    out = model(
        x,
        time=t,
        cross_context=cross_context,
        cross_context_mask=mask,
    )

    assert out.shape == x.shape


def test_conditioned_unet_cross_attention_blocks_are_inserted():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        condition_dim=32,
        attention_downsample_factors=(2, 4),
        attention_type="cross",
        cross_attention_dim=24,
        num_attention_heads=4,
    )

    attention_blocks = []
    for stage in model.encoder_stages:
        if isinstance(stage.attention, SpatialAttentionBlock):
            attention_blocks.append(stage.attention)

    for stage in model.decoder_stages:
        if isinstance(stage.stage.attention, SpatialAttentionBlock):
            attention_blocks.append(stage.stage.attention)

    assert attention_blocks
    assert all(block.attention_type == "cross" for block in attention_blocks)


def test_conditioned_unet_missing_time_raises():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        time_conditioning=True,
    )

    with pytest.raises(ValueError, match="time"):
        model(torch.randn(2, 4, 32, 32))


def test_conditioned_unet_unconfigured_time_raises():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        time_conditioning=False,
    )

    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))

    with pytest.raises(ValueError, match="time"):
        model(x, time=t)


def test_conditioned_unet_class_validation_raises():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        num_classes=10,
    )

    x = torch.randn(2, 4, 32, 32)

    with pytest.raises(ValueError, match="class_labels"):
        model(x)

    with pytest.raises(ValueError, match="class_labels"):
        model(x, class_labels=torch.ones(2, 1, dtype=torch.long))


def test_conditioned_unet_global_context_validation_raises():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=3,
        out_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        global_context_dim=12,
    )

    x = torch.randn(2, 3, 32, 32)

    with pytest.raises(ValueError, match="global_context"):
        model(x)

    with pytest.raises(ValueError, match="global_context"):
        model(x, global_context=torch.randn(2, 13))


def test_conditioned_unet_cross_context_validation_raises():
    model = ConditionedUNet(
        conv_dim=2,
        in_channels=4,
        out_channels=4,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        attention_downsample_factors=(2,),
        attention_type="cross",
        cross_attention_dim=24,
    )

    x = torch.randn(2, 4, 32, 32)

    with pytest.raises(ValueError, match="cross_context"):
        model(x)

    with pytest.raises(ValueError, match="cross_context"):
        model(x, cross_context=torch.randn(2, 24))

    with pytest.raises(ValueError, match="feature dimension"):
        model(x, cross_context=torch.randn(2, 5, 25))

    with pytest.raises(ValueError, match="mask"):
        model(
            x,
            cross_context=torch.randn(2, 5, 24),
            cross_context_mask=torch.ones(2, 4, dtype=torch.bool),
        )


def test_conditioned_unet_constructor_validation():
    with pytest.raises(ValueError, match="cross_attention_dim"):
        ConditionedUNet(
            conv_dim=2,
            in_channels=4,
            out_channels=4,
            stage_channels=[16, 32],
            attention_downsample_factors=(2,),
            attention_type="cross",
        )

    with pytest.raises(ValueError, match="class_dropout_prob"):
        ConditionedUNet(
            conv_dim=2,
            in_channels=4,
            out_channels=4,
            stage_channels=[16, 32],
            class_dropout_prob=0.1,
        )
