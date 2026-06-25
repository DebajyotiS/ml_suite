import pytest
import torch
import torch.nn as nn

from ml_suite.utils.activations import get_activation


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("relu", nn.ReLU),
        ("leakyrelu", nn.LeakyReLU),
        ("gelu", nn.GELU),
        ("sigmoid", nn.Sigmoid),
        ("tanh", nn.Tanh),
        ("softmax", nn.Softmax),
        ("silu", nn.SiLU),
        ("swish", nn.SiLU),
        ("mish", nn.Mish),
        ("identity", nn.Identity),
        ("none", nn.Identity),
    ],
)
def test_get_activation_returns_correct_type(name, expected_type):
    assert isinstance(get_activation(name), expected_type)


@pytest.mark.parametrize("name", ["relu", "RELU", "ReLU", "Relu"])
def test_get_activation_case_insensitive(name):
    assert isinstance(get_activation(name), nn.ReLU)


def test_get_activation_invalid_name_raises():
    with pytest.raises(ValueError, match="Unsupported activation function"):
        get_activation("banana")


def test_get_activation_returns_new_instance_each_call():
    a = get_activation("relu")
    b = get_activation("relu")
    assert a is not b


@pytest.mark.parametrize("name", ["relu", "gelu", "silu", "tanh", "identity"])
def test_get_activation_output_shape(name):
    act = get_activation(name)
    x = torch.randn(4, 16)
    out = act(x)
    assert out.shape == x.shape
