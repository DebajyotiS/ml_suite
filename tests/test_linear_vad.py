import pytest
import torch

from ml_suite.models.linear.vad import LinearVADInference, VADLinearBlock, VADState


# ---------------------------------------------------------------------------
# VADState
# ---------------------------------------------------------------------------


def test_vad_state_from_tensor_shape():
    mean = torch.randn(4, 16)
    state = VADState.from_tensor(mean)
    assert state.mu.shape == mean.shape
    assert state.var.shape == mean.shape


def test_vad_state_from_tensor_var_is_zero():
    mean = torch.randn(4, 16)
    state = VADState.from_tensor(mean)
    assert torch.all(state.var == 0)


def test_vad_state_from_tensor_mu_equals_input():
    mean = torch.randn(4, 16)
    state = VADState.from_tensor(mean)
    assert torch.equal(state.mu, mean)


def test_vad_state_add_shapes():
    a = VADState(mu=torch.randn(4, 16), var=torch.randn(4, 16))
    b = VADState(mu=torch.randn(4, 16), var=torch.randn(4, 16))
    c = a + b
    assert c.mu.shape == (4, 16)
    assert c.var.shape == (4, 16)


def test_vad_state_add_values():
    mu_a, var_a = torch.randn(4, 16), torch.randn(4, 16)
    mu_b, var_b = torch.randn(4, 16), torch.randn(4, 16)
    a = VADState(mu=mu_a, var=var_a)
    b = VADState(mu=mu_b, var=var_b)
    c = a + b
    assert torch.allclose(c.mu, mu_a + mu_b)
    assert torch.allclose(c.var, var_a + var_b)


def test_vad_state_add_rejects_non_vad_state():
    a = VADState(mu=torch.randn(4, 16), var=torch.randn(4, 16))
    with pytest.raises(ValueError, match="Can only add another VADState instance"):
        _ = a + torch.randn(4, 16)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# LinearVADInference
# ---------------------------------------------------------------------------


def test_linear_vad_inference_output_shape():
    net = LinearVADInference(input_dim=16, output_dim=8)
    x = torch.randn(4, 16)
    out = net(x)
    assert out.shape == (4, 8)


def test_linear_vad_inference_same_input_output_dim():
    net = LinearVADInference(input_dim=32, output_dim=32)
    x = torch.randn(2, 32)
    out = net(x)
    assert out.shape == (2, 32)


def test_linear_vad_inference_is_nn_module():
    import torch.nn as nn

    net = LinearVADInference(input_dim=8, output_dim=4)
    assert isinstance(net, nn.Module)


def test_linear_vad_inference_output_is_non_negative():
    """ReLU activations mean all outputs should be >= 0."""
    net = LinearVADInference(input_dim=8, output_dim=8)
    x = torch.randn(16, 8)
    out = net(x)
    assert torch.all(out >= 0)


# ---------------------------------------------------------------------------
# VADLinearBlock (stub guard)
# ---------------------------------------------------------------------------


def test_vad_linear_block_forward_raises_not_implemented():
    block = VADLinearBlock(input_dim=16, output_dim=16)
    state = VADState.from_tensor(torch.randn(2, 16))
    with pytest.raises(NotImplementedError):
        block(state)
