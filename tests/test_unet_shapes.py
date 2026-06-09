import pytest
import torch

from ml_suite.models.unet import UNet
from ml_suite.models.unet.attention import SpatialAttentionBlock


@pytest.mark.parametrize(
    "conv_dim,input_shape",
    [
        (1, (2, 3, 64)),
        (2, (2, 3, 64, 64)),
        (3, (2, 3, 16, 32, 32)),
    ],
)
@pytest.mark.parametrize("downsample_mode", ["stride", "pool"])
@pytest.mark.parametrize("upsample_mode", ["interpolate", "transpose"])
def test_unet_shapes_across_dims_and_sampling(
    conv_dim,
    input_shape,
    downsample_mode,
    upsample_mode,
):
    model = UNet(
        conv_dim=conv_dim,
        in_channels=3,
        out_channels=5,
        stage_channels=[8, 16, 32],
        blocks_per_stage=1,
        downsample_mode=downsample_mode,
        upsample_mode=upsample_mode,
        skip_mode="concat",
        norm_type="batch",
    )

    out = model(torch.randn(*input_shape))

    assert out.shape == (input_shape[0], 5) + input_shape[2:]


@pytest.mark.parametrize("skip_mode", ["concat", "add"])
def test_unet_skip_modes(skip_mode):
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=3,
        stage_channels=[8, 16, 32],
        blocks_per_stage=1,
        skip_mode=skip_mode,
    )

    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == x.shape


def test_unet_odd_shape_resize_policy_succeeds():
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=2,
        stage_channels=[8, 16, 32],
        blocks_per_stage=1,
        shape_policy="resize",
    )

    assert model(torch.randn(2, 3, 65, 67)).shape == (2, 2, 65, 67)


def test_unet_odd_shape_error_policy_raises():
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=2,
        stage_channels=[8, 16, 32],
        blocks_per_stage=1,
        shape_policy="error",
    )

    with pytest.raises(ValueError, match="spatial"):
        model(torch.randn(2, 3, 65, 67))


def test_unet_with_self_attention_forward_shape():
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=3,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        attention_downsample_factors=(2, 4),
        attention_locations="both",
        num_attention_heads=4,
    )

    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == x.shape


def test_unet_attention_blocks_are_inserted():
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=3,
        stage_channels=[16, 32, 64],
        blocks_per_stage=1,
        attention_downsample_factors=(2, 4),
        attention_locations="both",
        num_attention_heads=4,
    )

    encoder_count = sum(
        isinstance(stage.attention, SpatialAttentionBlock) for stage in model.encoder_stages
    )
    decoder_count = sum(
        isinstance(stage.stage.attention, SpatialAttentionBlock) for stage in model.decoder_stages
    )

    assert encoder_count >= 1
    assert decoder_count >= 1


def test_unet_output_activation():
    model = UNet(
        conv_dim=2,
        in_channels=3,
        out_channels=1,
        stage_channels=[8, 16],
        blocks_per_stage=1,
        output_activation="sigmoid",
    )

    out = model(torch.randn(2, 3, 32, 32))

    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_unet_validation_errors():
    with pytest.raises(ValueError, match="at least two"):
        UNet(2, 3, 1, [8], blocks_per_stage=1)

    with pytest.raises(ValueError, match="downsample_mode"):
        UNet(2, 3, 1, [8, 16], downsample_mode="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="upsample_mode"):
        UNet(2, 3, 1, [8, 16], upsample_mode="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="skip_mode"):
        UNet(2, 3, 1, [8, 16], skip_mode="bad")  # type: ignore[arg-type]
