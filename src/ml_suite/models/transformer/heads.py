"""Task heads for token and pooled transformer outputs."""

from __future__ import annotations

import torch
from torch import nn

from .pooling import TokenPooling
from .types import PoolingMode
from ...utils.activations import get_activation


def make_head_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    num_layers: int = 1,
    activation: str = "silu",
    dropout: float = 0.0,
) -> nn.Module:
    """Build a small MLP head."""
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive. Got {input_dim}.")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive. Got {output_dim}.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")
    if hidden_dim is not None and hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive. Got {hidden_dim}.")

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    if num_layers == 1:
        if dropout > 0.0:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
            )
        return nn.Linear(input_dim, output_dim)

    hidden = hidden_dim if hidden_dim is not None else input_dim
    layers: list[nn.Module] = []
    current_dim = input_dim

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class TokenwiseHead(nn.Module):
    """Map each token independently to an output dimension."""

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = make_head_mlp(
            input_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class PooledHead(nn.Module):
    """Pool tokens and map the pooled vector to an output dimension."""

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        pooling: PoolingMode = "mean",
        hidden_dim: int | None = None,
        num_layers: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooling = TokenPooling(mode=pooling, embedding_dim=embedding_dim)
        self.head = make_head_mlp(
            input_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pooling(tokens, mask=mask)
        return self.head(pooled)


class ClassificationHead(PooledHead):
    """Pooled classification head."""


class RegressionHead(PooledHead):
    """Pooled regression head."""
