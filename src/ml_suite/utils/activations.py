"""Activation function factory."""

import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """Return an activation function module by name.

    Args:
        activation: One of 'relu', 'leakyrelu', 'gelu', 'sigmoid', 'tanh',
            'softmax', 'silu'/'swish', 'mish', 'identity'/'none'.

    Returns:
        An nn.Module instance for the requested activation.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    match activation.lower():
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case "gelu":
            return nn.GELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "softmax":
            return nn.Softmax(dim=-1)
        case "silu" | "swish":
            return nn.SiLU()
        case "mish":
            return nn.Mish()
        case "identity" | "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported activation function: {activation}")
