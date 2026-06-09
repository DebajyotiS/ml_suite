import torch.nn as nn


def get_activation(activation: str):
    # Returns the activation function corresponding to the given string.
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
