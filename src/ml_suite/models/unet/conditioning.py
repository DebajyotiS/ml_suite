"""Global conditioning modules for U-Nets."""

import torch
from torch import nn

from ml_suite.utils.conditioning import SinusoidalTimeEmbedding, TimeEmbeddingMLP
from .types import TimeEmbeddingType

__all__ = ["SinusoidalTimeEmbedding", "TimeEmbeddingMLP", "ConditioningBuilder"]


class ConditioningBuilder(nn.Module):
    """Builds a single FiLM conditioning vector from time, class, and context.

    The returned tensor always has shape ``(B, condition_dim)``.
    """

    def __init__(
        self,
        condition_dim: int,
        time_conditioning: bool = False,
        time_embedding_type: TimeEmbeddingType = "sinusoidal",
        num_classes: int | None = None,
        class_dropout_prob: float = 0.0,
        global_context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive. Got {condition_dim}.")
        if time_embedding_type not in ("sinusoidal", "learned"):
            raise ValueError(
                f"time_embedding_type must be 'sinusoidal' or 'learned'. Got {time_embedding_type}."
            )
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"num_classes must be positive. Got {num_classes}.")
        if class_dropout_prob < 0.0 or class_dropout_prob >= 1.0:
            raise ValueError(f"class_dropout_prob must be in [0, 1). Got {class_dropout_prob}.")
        if class_dropout_prob > 0.0 and num_classes is None:
            raise ValueError("class_dropout_prob requires num_classes.")
        if global_context_dim is not None and global_context_dim <= 0:
            raise ValueError(f"global_context_dim must be positive. Got {global_context_dim}.")

        self.condition_dim = condition_dim
        self.time_conditioning = time_conditioning
        self.time_embedding_type = time_embedding_type
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.global_context_dim = global_context_dim

        self.time_embedding = (
            TimeEmbeddingMLP(condition_dim, time_embedding_type) if time_conditioning else None
        )

        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes + 1, condition_dim)
            self.null_class_index = num_classes
        else:
            self.class_embedding = None
            self.null_class_index = None

        self.global_context_projection = (
            nn.Linear(global_context_dim, condition_dim) if global_context_dim is not None else None
        )

    @property
    def is_enabled(self) -> bool:
        return (
            self.time_conditioning
            or self.num_classes is not None
            or self.global_context_dim is not None
        )

    def _apply_class_dropout(self, class_labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.class_dropout_prob == 0.0:
            return class_labels
        assert self.null_class_index is not None
        labels = class_labels.clone()
        dropout_mask = torch.rand(labels.shape, device=labels.device) < self.class_dropout_prob
        labels[dropout_mask] = self.null_class_index
        return labels

    def validate_inputs(
        self,
        batch_size: int,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
    ) -> None:
        if self.time_conditioning and time is None:
            raise ValueError("time must be provided when time_conditioning=True.")
        if not self.time_conditioning and time is not None:
            raise ValueError("time was provided, but time_conditioning=False.")
        if time is not None and time.shape[0] != batch_size:
            raise ValueError(
                f"time batch size ({time.shape[0]}) must match input batch size ({batch_size})."
            )

        if self.num_classes is not None and class_labels is None:
            raise ValueError("class_labels must be provided when num_classes is set.")
        if self.num_classes is None and class_labels is not None:
            raise ValueError("class_labels was provided, but num_classes is None.")
        if class_labels is not None:
            if class_labels.ndim != 1:
                raise ValueError(
                    f"class_labels must have shape (batch,). Got {class_labels.shape}."
                )
            if class_labels.shape[0] != batch_size:
                raise ValueError(
                    f"class_labels batch size ({class_labels.shape[0]}) must match "
                    f"input batch size ({batch_size})."
                )
            if class_labels.min().item() < 0 or class_labels.max().item() > self.num_classes:
                raise ValueError(
                    f"class_labels must be in [0, {self.num_classes}], where "
                    f"{self.num_classes} is the optional null class index."
                )

        if self.global_context_dim is not None and global_context is None:
            raise ValueError("global_context must be provided when global_context_dim is set.")
        if self.global_context_dim is None and global_context is not None:
            raise ValueError("global_context was provided, but global_context_dim is None.")
        if global_context is not None:
            if global_context.ndim != 2:
                raise ValueError(
                    f"global_context must have shape (batch, dim). Got {global_context.shape}."
                )
            if global_context.shape[0] != batch_size:
                raise ValueError(
                    f"global_context batch size ({global_context.shape[0]}) must match "
                    f"input batch size ({batch_size})."
                )
            if global_context.shape[1] != self.global_context_dim:
                raise ValueError(
                    f"global_context feature dimension must be {self.global_context_dim}. "
                    f"Got {global_context.shape[1]}."
                )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the combined FiLM conditioning vector.

        All enabled conditioning sources are projected to condition_dim and summed.

        Args:
            batch_size: Number of samples in the batch.
            device: Target device for the output tensor.
            dtype: Target dtype for the output tensor.
            time: Timestep tensor of shape (batch,). Required when time_conditioning=True.
            class_labels: Integer class indices of shape (batch,). Required when num_classes is set.
            global_context: Global context of shape (batch, global_context_dim).
                Required when global_context_dim is set.

        Returns:
            Conditioning vector of shape (batch, condition_dim).
        """
        self.validate_inputs(batch_size, time, class_labels, global_context)

        condition = torch.zeros(batch_size, self.condition_dim, device=device, dtype=dtype)

        if self.time_embedding is not None:
            assert time is not None
            condition = condition + self.time_embedding(time).to(device=device, dtype=dtype)

        if self.class_embedding is not None:
            assert class_labels is not None
            labels = self._apply_class_dropout(class_labels.long())
            condition = condition + self.class_embedding(labels).to(device=device, dtype=dtype)

        if self.global_context_projection is not None:
            assert global_context is not None
            condition = condition + self.global_context_projection(global_context).to(
                device=device,
                dtype=dtype,
            )

        return condition
