"""Global conditioning utilities for token transformers."""

from __future__ import annotations

import torch
from torch import nn

from ml_suite.utils.conditioning import SinusoidalTimeEmbedding, TimeEmbeddingMLP

from .types import TimeEmbeddingType

__all__ = [
    "SinusoidalTimeEmbedding",
    "TimeEmbeddingMLP",
    "TransformerConditioningBuilder",
    "ConditionTokenProjector",
]


class TransformerConditioningBuilder(nn.Module):
    """Build a single global conditioning vector."""

    def __init__(
        self,
        embedding_dim: int,
        time_conditioning: bool = False,
        time_embedding_type: TimeEmbeddingType = "sinusoidal",
        num_classes: int | None = None,
        class_dropout_prob: float = 0.0,
        global_context_dim: int | None = None,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"num_classes must be positive. Got {num_classes}.")
        if class_dropout_prob < 0.0 or class_dropout_prob >= 1.0:
            raise ValueError(f"class_dropout_prob must be in [0, 1). Got {class_dropout_prob}.")
        if class_dropout_prob > 0.0 and num_classes is None:
            raise ValueError("class_dropout_prob requires num_classes.")
        if global_context_dim is not None and global_context_dim <= 0:
            raise ValueError(f"global_context_dim must be positive. Got {global_context_dim}.")

        self.embedding_dim = embedding_dim
        self.time_conditioning = time_conditioning
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.global_context_dim = global_context_dim

        if time_conditioning:
            self.time_embedding = TimeEmbeddingMLP(
                embedding_dim=embedding_dim,
                embedding_type=time_embedding_type,
            )

        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
            self.null_class_index = num_classes

        if global_context_dim is not None:
            self.global_context_projection = nn.Linear(global_context_dim, embedding_dim)

    def _apply_class_dropout(self, class_labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.class_dropout_prob == 0.0:
            return class_labels

        labels = class_labels.clone()
        dropout_mask = torch.rand(labels.shape, device=labels.device) < self.class_dropout_prob
        labels[dropout_mask] = self.null_class_index
        return labels

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a conditioning vector shaped (batch, embedding_dim)."""
        parts: list[torch.Tensor] = []

        if self.time_conditioning:
            if time is None:
                raise ValueError("time must be provided when time_conditioning=True.")
            if time.shape[0] != batch_size:
                raise ValueError("time batch size must match input batch size.")
            parts.append(self.time_embedding(time).to(device=device, dtype=dtype))
        elif time is not None:
            raise ValueError("time was provided, but time_conditioning=False.")

        if self.num_classes is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided when num_classes is set.")
            if class_labels.ndim != 1 or class_labels.shape[0] != batch_size:
                raise ValueError("class_labels must have shape (batch,).")
            labels = self._apply_class_dropout(class_labels.long())
            parts.append(self.class_embedding(labels).to(device=device, dtype=dtype))
        elif class_labels is not None:
            raise ValueError("class_labels was provided, but num_classes is None.")

        if self.global_context_dim is not None:
            if global_context is None:
                raise ValueError("global_context must be provided when global_context_dim is set.")
            if global_context.shape != (batch_size, self.global_context_dim):
                raise ValueError(
                    f"global_context must have shape {(batch_size, self.global_context_dim)}. "
                    f"Got {global_context.shape}."
                )
            parts.append(
                self.global_context_projection(global_context).to(device=device, dtype=dtype)
            )
        elif global_context is not None:
            raise ValueError("global_context was provided, but global_context_dim is None.")

        if parts:
            return torch.stack(parts).sum(0)
        return torch.zeros(batch_size, self.embedding_dim, device=device, dtype=dtype)

    def has_conditioning(self) -> bool:
        """Return True if any conditioning source is configured."""
        return bool(self.time_conditioning or self.num_classes or self.global_context_dim)


class ConditionTokenProjector(nn.Module):
    """Project a global context vector into one or more condition tokens."""

    def __init__(
        self,
        context_dim: int,
        embedding_dim: int,
        num_tokens: int = 1,
    ) -> None:
        super().__init__()

        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive. Got {context_dim}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive. Got {num_tokens}.")

        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.projection = nn.Linear(context_dim, num_tokens * embedding_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Project a global context vector into one or more condition tokens.

        Args:
            context: Global context tensor of shape (batch, context_dim).

        Returns:
            Condition tokens of shape (batch, num_tokens, embedding_dim).
        """
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape (batch, {self.context_dim}). Got {context.shape}."
            )

        batch = context.shape[0]
        return self.projection(context).view(batch, self.num_tokens, self.embedding_dim)
