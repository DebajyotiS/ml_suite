"""Token pooling modules."""

from __future__ import annotations

import torch
from torch import nn

from .types import PoolingMode
from .utils import last_valid_indices, validate_mask, validate_token_tensor


class TokenPooling(nn.Module):
    """Pool token tensors into a vector."""

    def __init__(
        self,
        mode: PoolingMode = "mean",
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()

        if mode not in ("mean", "max", "cls", "last", "attention"):
            raise ValueError(f"Unsupported pooling mode: {mode}.")
        if mode == "attention" and embedding_dim is None:
            raise ValueError("embedding_dim is required for attention pooling.")

        self.mode = mode
        self.embedding_dim = embedding_dim
        self.attention_score = nn.Linear(embedding_dim, 1) if mode == "attention" else None

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool tokens shaped (batch, tokens, dim)."""
        validate_token_tensor(tokens)

        if mask is not None:
            validate_mask(mask, tokens)

        if self.mode == "cls":
            return tokens[:, 0]

        if self.mode == "last":
            if mask is None:
                return tokens[:, -1]
            indices = last_valid_indices(mask)
            return tokens[torch.arange(tokens.shape[0], device=tokens.device), indices]

        if self.mode == "mean":
            if mask is None:
                return tokens.mean(dim=1)
            weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (tokens * weights).sum(dim=1) / denom

        if self.mode == "max":
            if mask is None:
                return tokens.max(dim=1).values
            masked = tokens.masked_fill(~mask[:, :, None], float("-inf"))
            return masked.max(dim=1).values

        if self.mode == "attention":
            scores = self.attention_score(tokens).squeeze(-1)  # type: ignore[union-attr]
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(tokens.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (tokens * weights).sum(dim=1)

        raise RuntimeError("Unreachable pooling mode.")
