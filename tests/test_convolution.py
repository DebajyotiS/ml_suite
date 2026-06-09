from typing import Literal

import pytest
import torch
from torch import nn

from ml_suite.models.convolution import (
    ConditionedConvBlock,
    ConditionedConvNet,
    ConvBlock,
    ConvNet,
)


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        (1, (2, 8, 32)),
        (2, (2, 8, 32, 32)),
        (3, (2, 8, 16, 32, 32)),
    ],
)
@pytest.mark.parametrize("norm_type", ["batch", "group", "layer", None])
def test_conv_block_dimensions_and_norms(
    dim: int,
    input_shape: tuple[int, ...],
    norm_type: Literal["batch", "group", "layer"] | None,
):
    """ConvBlock should handle 1D, 2D, and 3D inputs across all norm modes."""
    out_channels = 16

    block = ConvBlock(
        input_channels=8,
        output_channels=out_channels,
        conv_dim=dim,
        kernel_size=3,
        stride=1,
        padding=1,
        norm_type=norm_type,
        num_groups=4,
    )

    x = torch.randn(*input_shape)
    out = block(x)

    expected_shape = (input_shape[0], out_channels) + input_shape[2:]
    assert out.shape == expected_shape


def test_conv_block_residual_matching():
    """Residual addition should work when input and output shapes match."""
    x = torch.randn(2, 8, 16, 16)

    block = ConvBlock(
        input_channels=8,
        output_channels=8,
        conv_dim=2,
        do_residual=True,
    )

    out = block(x)

    assert out.shape == x.shape


def test_conv_block_invalid_residual_channel_mismatch_raises_error():
    """Residual blocks should reject channel mismatches at initialization."""
    with pytest.raises(ValueError, match="Residual connection requires input_channels"):
        ConvBlock(
            input_channels=8,
            output_channels=16,
            conv_dim=2,
            do_residual=True,
        )


def test_conv_block_residual_shape_mismatch_raises_error_at_forward():
    """Residual blocks should reject spatial shape mismatches at forward time."""
    x = torch.randn(2, 8, 16, 16)

    block = ConvBlock(
        input_channels=8,
        output_channels=8,
        conv_dim=2,
        stride=2,
        padding=1,
        do_residual=True,
    )

    with pytest.raises(ValueError, match="Residual connection shape mismatch"):
        block(x)


def test_conv_block_group_norm_divisibility_guard():
    """GroupNorm should require output_channels divisible by num_groups."""
    with pytest.raises(ValueError, match="must be divisible by"):
        ConvBlock(
            input_channels=4,
            output_channels=10,
            conv_dim=2,
            norm_type="group",
            num_groups=4,
        )


def test_conv_block_invalid_dimension_raises_error():
    """Invalid convolutional dimensionality should be rejected."""
    with pytest.raises(ValueError, match="Unsupported conv_dim"):
        ConvBlock(
            input_channels=4,
            output_channels=4,
            conv_dim=4,
        )


def test_conv_block_invalid_norm_type_raises_error():
    """Invalid normalization mode should be rejected."""
    with pytest.raises(ValueError, match="Unsupported norm_type"):
        ConvBlock(
            input_channels=4,
            output_channels=4,
            conv_dim=2,
            norm_type="instance",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "input_channels, output_channels",
    [
        (0, 8),
        (8, 0),
        (-1, 8),
        (8, -1),
    ],
)
def test_conv_block_nonpositive_channels_raise_error(
    input_channels: int,
    output_channels: int,
):
    """Channel counts should be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        ConvBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            conv_dim=2,
        )


def test_conv_block_strings():
    """__repr__ and __str__ should include useful architectural metadata."""
    block = ConvBlock(
        input_channels=3,
        output_channels=12,
        conv_dim=2,
        norm_type="layer",
    )

    repr_str = repr(block)
    str_str = str(block)

    assert "input_channels=3" in repr_str
    assert "output_channels=12" in repr_str
    assert "conv_dim=2" in repr_str
    assert "GroupNorm" in str_str
    assert "Conv2D" in str_str


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        (1, (2, 8, 32)),
        (2, (2, 8, 32, 32)),
        (3, (2, 8, 16, 32, 32)),
    ],
)
@pytest.mark.parametrize("norm_type", ["batch", "group", "layer", None])
def test_conditioned_conv_block_dimensions_and_norms(
    dim: int,
    input_shape: tuple[int, ...],
    norm_type: Literal["batch", "group", "layer"] | None,
):
    """ConditionedConvBlock should handle 1D, 2D, and 3D inputs across norm modes."""
    batch_size = input_shape[0]
    context_dim = 11
    out_channels = 16

    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=out_channels,
        context_dim=context_dim,
        conv_dim=dim,
        kernel_size=3,
        stride=1,
        padding=1,
        norm_type=norm_type,
        num_groups=4,
    )

    x = torch.randn(*input_shape)
    context = torch.randn(batch_size, context_dim)

    out = block(x, context)

    expected_shape = (batch_size, out_channels) + input_shape[2:]
    assert out.shape == expected_shape


def test_conditioned_conv_block_film_projection_starts_as_identity():
    """Zero-initialized FiLM should make the block initially independent of context."""
    torch.manual_seed(0)

    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=8,
        context_dim=5,
        conv_dim=2,
        norm_type=None,
        do_residual=True,
    )

    x = torch.randn(2, 8, 16, 16)
    context_a = torch.randn(2, 5)
    context_b = torch.randn(2, 5)

    out_a = block(x, context_a)
    out_b = block(x, context_b)

    assert torch.allclose(
        block.context_projection.weight, torch.zeros_like(block.context_projection.weight)
    )
    assert torch.allclose(
        block.context_projection.bias, torch.zeros_like(block.context_projection.bias)
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_conditioned_conv_block_rejects_bad_context_rank():
    """Context must be a rank-2 tensor."""
    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=8,
        context_dim=5,
        conv_dim=2,
    )

    x = torch.randn(2, 8, 16, 16)
    context = torch.randn(2, 5, 1)

    with pytest.raises(ValueError, match="context must have shape"):
        block(x, context)


def test_conditioned_conv_block_rejects_context_batch_mismatch():
    """Context batch size must match input batch size."""
    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=8,
        context_dim=5,
        conv_dim=2,
    )

    x = torch.randn(2, 8, 16, 16)
    context = torch.randn(3, 5)

    with pytest.raises(ValueError, match="context batch size"):
        block(x, context)


def test_conditioned_conv_block_rejects_context_dim_mismatch():
    """Context feature dimension must match context_dim."""
    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=8,
        context_dim=5,
        conv_dim=2,
    )

    x = torch.randn(2, 8, 16, 16)
    context = torch.randn(2, 6)

    with pytest.raises(ValueError, match="context feature dimension"):
        block(x, context)


def test_conditioned_conv_block_invalid_context_dim_raises_error():
    """context_dim must be positive."""
    with pytest.raises(ValueError, match="context_dim must be positive"):
        ConditionedConvBlock(
            input_channels=8,
            output_channels=8,
            context_dim=0,
            conv_dim=2,
        )


def test_conditioned_conv_block_residual_shape_mismatch_raises_error_at_forward():
    """Conditioned residual blocks should reject spatial shape mismatches."""
    block = ConditionedConvBlock(
        input_channels=8,
        output_channels=8,
        context_dim=5,
        conv_dim=2,
        stride=2,
        padding=1,
        do_residual=True,
    )

    x = torch.randn(2, 8, 16, 16)
    context = torch.randn(2, 5)

    with pytest.raises(ValueError, match="Residual connection shape mismatch"):
        block(x, context)


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        (1, (2, 3, 64)),
        (2, (2, 3, 64, 64)),
        (3, (2, 3, 32, 64, 64)),
    ],
)
@pytest.mark.parametrize("downsample_mode", ["stride", "pool"])
def test_conv_net_feature_extraction_shapes(
    dim: int,
    input_shape: tuple[int, ...],
    downsample_mode: Literal["stride", "pool"],
):
    """ConvNet should return spatial features when num_classes=None."""
    stage_channels = [16, 32, 64]

    net = ConvNet(
        conv_dim=dim,
        in_channels=3,
        stage_channels=stage_channels,
        blocks_per_stage=2,
        downsample_mode=downsample_mode,
        num_classes=None,
    )

    x = torch.randn(*input_shape)
    out = net(x)

    expected_spatial_shape = tuple(size // 4 for size in input_shape[2:])

    assert out.shape[0] == input_shape[0]
    assert out.shape[1] == stage_channels[-1]
    assert out.shape[2:] == expected_spatial_shape


@pytest.mark.parametrize("pool_mode", ["avg", "max", "cat_avg_max"])
def test_conv_net_classification_head_shapes(
    pool_mode: Literal["avg", "max", "cat_avg_max"],
):
    """ConvNet classification head should return shape (batch, num_classes)."""
    num_classes = 10

    net = ConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=[1, 2],
        global_pool_mode=pool_mode,
        num_classes=num_classes,
    )

    x = torch.randn(2, 3, 32, 32)
    out = net(x)

    assert out.shape == (2, num_classes)


def test_conv_net_asymmetric_stage_blocks_stride_mode():
    """ConvNet should preserve requested block counts in stride mode."""
    net = ConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32, 64],
        blocks_per_stage=[1, 3, 2],
        downsample_mode="stride",
    )

    assert len(net.stages[0]) == 1
    assert len(net.stages[1]) == 3
    assert len(net.stages[2]) == 2


def test_conv_net_pool_mode_transition_adds_pool_module():
    """Pool mode should add an explicit MaxPool layer at transition stages."""
    net = ConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=[1, 1],
        downsample_mode="pool",
    )

    assert len(net.stages[0]) == 1
    assert len(net.stages[1]) == 2
    assert isinstance(net.stages[1][0], ConvBlock)
    assert isinstance(net.stages[1][1], nn.MaxPool2d)


def test_conv_net_block_length_mismatch_raises_error():
    """blocks_per_stage sequence length must match stage_channels length."""
    with pytest.raises(ValueError, match="Length of blocks_per_stage"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=[1, 2, 3],
        )


def test_conv_net_empty_stage_channels_raises_error():
    """stage_channels must not be empty."""
    with pytest.raises(ValueError, match="stage_channels must contain"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[],
            blocks_per_stage=1,
        )


@pytest.mark.parametrize(
    "blocks_per_stage",
    [
        0,
        [1, 0],
        [0, 1],
    ],
)
def test_conv_net_nonpositive_blocks_per_stage_raises_error(
    blocks_per_stage: int | list[int],
):
    """Every stage must contain at least one block."""
    with pytest.raises(ValueError, match="Every stage must contain at least one block"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=blocks_per_stage,
        )


def test_conv_net_invalid_downsample_mode_raises_error():
    """Invalid downsample mode should be rejected."""
    with pytest.raises(ValueError, match="downsample_mode must be"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=1,
            downsample_mode="blurpool",  # type: ignore[arg-type]
        )


def test_conv_net_invalid_global_pool_mode_raises_error():
    """Invalid global pooling mode should be rejected."""
    with pytest.raises(ValueError, match="global_pool_mode must be"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=1,
            global_pool_mode="median",  # type: ignore[arg-type]
            num_classes=10,
        )


def test_conv_net_invalid_num_classes_raises_error():
    """num_classes must be positive when provided."""
    with pytest.raises(ValueError, match="num_classes must be positive"):
        ConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=1,
            num_classes=0,
        )


def test_conv_net_receptive_field_runner(capsys: pytest.CaptureFixture[str]):
    """The analytical receptive-field utility should print tabular output."""
    net = ConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        downsample_mode="stride",
    )

    net.print_receptive_field()
    captured = capsys.readouterr()

    assert "Layer / Module" in captured.out
    assert "Cumulative RF" in captured.out
    assert "Stem" in captured.out
    assert "Stage 2" in captured.out


def test_conv_net_strings():
    """ConvNet string methods should expose stage and head metadata."""
    net = ConvNet(
        conv_dim=2,
        in_channels=1,
        stage_channels=[16],
        blocks_per_stage=1,
        num_classes=5,
    )

    repr_str = repr(net)
    str_str = str(net)

    assert "stage_channels=[16]" in repr_str
    assert "Global Pooling:" in str_str
    assert "Head: Linear" in str_str


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        (1, (2, 3, 64)),
        (2, (2, 3, 64, 64)),
        (3, (2, 3, 32, 64, 64)),
    ],
)
@pytest.mark.parametrize("downsample_mode", ["stride", "pool"])
def test_conditioned_conv_net_feature_extraction_shapes(
    dim: int,
    input_shape: tuple[int, ...],
    downsample_mode: Literal["stride", "pool"],
):
    """ConditionedConvNet should return conditioned spatial features when num_classes=None."""
    context_dim = 7
    stage_channels = [16, 32, 64]

    net = ConditionedConvNet(
        conv_dim=dim,
        in_channels=3,
        stage_channels=stage_channels,
        blocks_per_stage=2,
        context_dim=context_dim,
        downsample_mode=downsample_mode,
        num_classes=None,
    )

    x = torch.randn(*input_shape)
    context = torch.randn(input_shape[0], context_dim)

    out = net(x, context)

    expected_spatial_shape = tuple(size // 4 for size in input_shape[2:])

    assert out.shape[0] == input_shape[0]
    assert out.shape[1] == stage_channels[-1]
    assert out.shape[2:] == expected_spatial_shape


@pytest.mark.parametrize("pool_mode", ["avg", "max", "cat_avg_max"])
def test_conditioned_conv_net_classification_head_shapes(
    pool_mode: Literal["avg", "max", "cat_avg_max"],
):
    """ConditionedConvNet classification head should return shape (batch, num_classes)."""
    context_dim = 7
    num_classes = 10

    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=[1, 2],
        context_dim=context_dim,
        global_pool_mode=pool_mode,
        num_classes=num_classes,
    )

    x = torch.randn(2, 3, 32, 32)
    context = torch.randn(2, context_dim)

    out = net(x, context)

    assert out.shape == (2, num_classes)


def test_conditioned_conv_net_uses_conditioned_blocks_in_stages():
    """ConditionedConvNet should directly build conditioned stage blocks."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=2,
        context_dim=7,
        downsample_mode="stride",
    )

    for stage in net.stages:
        for module in stage:
            assert isinstance(module, ConditionedConvBlock)


def test_conditioned_conv_net_stem_is_unconditioned():
    """The stem should remain a plain ConvBlock."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        context_dim=7,
    )

    assert isinstance(net.stem, ConvBlock)
    assert not isinstance(net.stem, ConditionedConvBlock)


def test_conditioned_conv_net_pool_mode_contains_pool_modules():
    """ConditionedConvNet with pool downsampling should include explicit pool modules."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=[1, 1],
        context_dim=7,
        downsample_mode="pool",
    )

    assert len(net.stages[0]) == 1
    assert len(net.stages[1]) == 2
    assert isinstance(net.stages[1][0], ConditionedConvBlock)
    assert isinstance(net.stages[1][1], nn.MaxPool2d)


def test_conditioned_conv_net_invalid_context_dim_raises_error():
    """ConditionedConvNet should reject non-positive context_dim."""
    with pytest.raises(ValueError, match="context_dim must be positive"):
        ConditionedConvNet(
            conv_dim=2,
            in_channels=3,
            stage_channels=[16, 32],
            blocks_per_stage=1,
            context_dim=0,
        )


def test_conditioned_conv_net_forward_rejects_context_batch_mismatch():
    """ConditionedConvNet should reject context tensors with wrong batch size."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        context_dim=7,
    )

    x = torch.randn(2, 3, 32, 32)
    context = torch.randn(3, 7)

    with pytest.raises(ValueError, match="context batch size"):
        net(x, context)


def test_conditioned_conv_net_forward_rejects_context_dim_mismatch():
    """ConditionedConvNet should reject context tensors with wrong feature dimension."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=3,
        stage_channels=[16, 32],
        blocks_per_stage=1,
        context_dim=7,
    )

    x = torch.randn(2, 3, 32, 32)
    context = torch.randn(2, 8)

    with pytest.raises(ValueError, match="context feature dimension"):
        net(x, context)


def test_conditioned_conv_net_strings():
    """ConditionedConvNet string methods should expose context and head metadata."""
    net = ConditionedConvNet(
        conv_dim=2,
        in_channels=1,
        stage_channels=[16],
        blocks_per_stage=1,
        context_dim=7,
        num_classes=5,
    )

    repr_str = repr(net)
    str_str = str(net)

    assert "ConditionedConvNet" in repr_str
    assert "context_dim=7" in repr_str
    assert "stage_channels=[16]" in repr_str
    assert "Global Pooling:" in str_str
    assert "Head: Linear" in str_str
