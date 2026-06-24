"""Variational Adaptive Dropout (VAD) modules — experimental, not yet implemented."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class VADState:
    mu: torch.Tensor
    var: torch.Tensor

    @staticmethod
    def from_tensor(mean: torch.Tensor) -> VADState:
        return VADState(mu=mean, var=torch.zeros_like(mean))

    def __add__(self, other: VADState) -> VADState:
        if not isinstance(other, VADState):
            raise ValueError("Can only add another VADState instance.")
        return VADState(mu=self.mu + other.mu, var=self.var + other.var)


class LinearVADInference(nn.Module):
    """Small MLP for inferring per-layer adaptive (input-dependent) dropout scale."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)


class VADLinearBlock(nn.Module):
    """Placeholder for a VAD-equipped linear block. Not yet implemented."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: int = 0,
        context_injection: Literal["concat", "add", "multiply", "film", "cross_attn"] = "concat",
        activation: str = "silu",
        do_residual: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.vad_inference = LinearVADInference(input_dim, output_dim)
        self.p = p

    def forward(self, x: VADState, context: torch.Tensor | None = None) -> VADState:
        raise NotImplementedError(
            "VADLinearBlock forward pass is not implemented yet."
        )
