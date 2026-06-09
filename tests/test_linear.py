import pytest
import torch
import torch.nn as nn

from ml_suite.models.linear import LinearBlock, MLP


@pytest.mark.parametrize("injection", ["concat", "add", "multiply", "film", "cross_attn"])
def test_linear_block_forward_shapes(injection: str):
    """Verify that all context injection mechanisms output correct tensor shapes."""
    batch_size = 4
    input_dim = 16
    output_dim = 32
    context_dim = 8

    block = LinearBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        context_dim=context_dim,
        context_injection=injection,  # type: ignore
        activation="silu",
    )

    x = torch.randn(batch_size, input_dim)
    context = torch.randn(batch_size, context_dim)

    out = block(x, context)
    assert out.shape == (batch_size, output_dim)


def test_linear_block_no_context_shape():
    """Verify that LinearBlock functions correctly when context_dim is 0."""
    batch_size = 3
    input_dim = 12
    output_dim = 24

    block = LinearBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        context_dim=0,
    )

    x = torch.randn(batch_size, input_dim)
    out = block(x)
    assert out.shape == (batch_size, output_dim)


def test_linear_block_high_dimensional_input():
    """Verify that LinearBlock preserves extra trailing dimensions (e.g., sequence length)."""
    shape = (2, 5, 4, 16)  # (Batch, Seq, Extra, Input_Dim)
    context_shape = (2, 5, 4, 8)

    block = LinearBlock(input_dim=16, output_dim=32, context_dim=8, context_injection="film")

    x = torch.randn(*shape)
    context = torch.randn(*context_shape)

    out = block(x, context)
    assert out.shape == (2, 5, 4, 32)


def test_linear_block_missing_context_raises_error():
    """Assert forward pass crashes when a required context tensor is omitted."""
    block = LinearBlock(input_dim=16, output_dim=16, context_dim=8)
    x = torch.randn(2, 16)

    with pytest.raises(ValueError, match="Context tensor is required but not provided."):
        block(x, context=None)


def test_linear_block_invalid_residual_dimensions():
    """Assert initialization checks block residual shape mismatches."""
    with pytest.raises(ValueError, match="Residual connection requires input_dim"):
        LinearBlock(input_dim=16, output_dim=32, do_residual=True)


def test_film_identity_behavior():
    """FiLM conditioning with zero context should yield unconditioned outputs."""
    input_dim = 8
    output_dim = 8
    context_dim = 4

    block = LinearBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        context_dim=context_dim,
        context_injection="film",
        activation="silu",
    )

    # Freeze weights for explicit matching
    nn.init.ones_(block.net.weight)
    nn.init.zeros_(block.net.bias)
    nn.init.zeros_(block.film_scale.weight)
    nn.init.zeros_(block.film_scale.bias)
    nn.init.zeros_(block.film_shift.weight)
    nn.init.zeros_(block.film_shift.bias)

    x = torch.randn(2, input_dim)
    zero_context = torch.zeros(2, context_dim)

    # Scale becomes 0 + 1 = 1, shift becomes 0
    out_conditioned = block(x, zero_context)
    expected = torch.nn.functional.silu(block.net(x))

    assert torch.allclose(out_conditioned, expected, atol=1e-6)


def test_linear_block_strings():
    """Verify __repr__ and __str__ do not crash and format parameters accurately."""
    block = LinearBlock(input_dim=10, output_dim=20, context_dim=5, context_injection="add")

    repr_str = repr(block)
    str_str = str(block)

    assert "input_dim=10" in repr_str
    assert "output_dim=20" in repr_str
    assert "context_dim=5" in repr_str
    assert "Linear" in str_str


def test_mlp_uniform_hidden_forward_shape():
    """Verify standard uniform dimension propagation through an MLP pipeline."""
    batch_size = 4
    input_dim = 8
    hidden_dim = 16
    num_layers = 3
    context_dim = 6

    mlp = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_dim=context_dim,
        context_injection="concat",
    )

    x = torch.randn(batch_size, input_dim)
    context = torch.randn(batch_size, context_dim)

    out = mlp(x, context)
    assert out.shape == (batch_size, hidden_dim)


def test_mlp_custom_hidden_layers_forward_shape():
    """Verify MLP shape routing when custom hidden_layers lists are provided."""
    batch_size = 2
    input_dim = 10
    hidden_dim = 5
    num_layers = 4
    hidden_layers = [
        20,
        30,
        15,
    ]

    mlp = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        hidden_layers=hidden_layers,
        context_dim=0,
    )

    x = torch.randn(batch_size, input_dim)
    out = mlp(x)
    assert out.shape == (batch_size, hidden_dim)


def test_mlp_selective_residual_routing():
    """Verify residual flags are systematically disabled if block layer input/output dimensions diverge."""
    mlp = MLP(
        input_dim=16,
        hidden_dim=64,
        num_layers=3,
        hidden_layers=[16, 64],
        do_residual=True,
    )

    assert mlp.blocks[0].do_residual is True  # Matches 16 -> 16
    assert mlp.blocks[1].do_residual is False  # Mismatch 16 -> 64
    assert mlp.blocks[2].do_residual is True  # Matches 64 -> 64


def test_mlp_invalid_num_layers():
    """Verify initialization checks for zero or negative layer structures."""
    with pytest.raises(ValueError, match="num_layers must be at least 1."):
        MLP(input_dim=10, hidden_dim=10, num_layers=0)


def test_mlp_hidden_layers_mismatch_raises_error():
    """Assert initialization crashes if hidden_layers list length does not match num_layers - 1."""
    with pytest.raises(ValueError, match="Length of hidden_layers"):
        MLP(input_dim=10, hidden_dim=20, num_layers=3, hidden_layers=[32])  # Needs 2 layers


def test_mlp_activation_list_mismatch_raises_error():
    """Assert initialization crashes if activation_list length does not equal num_layers."""
    with pytest.raises(ValueError, match="Length of activation_list"):
        MLP(input_dim=10, hidden_dim=20, num_layers=2, activation_list=["relu"])  # Needs 2 entries


def test_mlp_strings():
    """Verify MLP log generation features clear sub-block structure outputs."""
    mlp = MLP(input_dim=10, hidden_dim=20, num_layers=2, context_dim=4)

    repr_str = repr(mlp)
    str_str = str(mlp)

    assert "input_dim=10" in repr_str
    assert "hidden_dim=20" in repr_str
    assert "Blocks Stack:" in str_str
    assert "[0] └──" in str_str
