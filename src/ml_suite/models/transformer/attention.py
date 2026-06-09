"""Multi-head self-attention and cross-attention primitives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .positional import RotaryEmbedding, apply_rope_to_tensor
from .types import PositionalEncodingMode
from .utils import normalize_positional_encoding_mode, validate_mask, validate_token_tensor


def _get_device_type(device: torch.device | str) -> str:
    """Return the device type string ('cuda', 'mps', or 'cpu')."""
    return torch.device(device).type


def _sdpa_is_flash_eligible(q: torch.Tensor) -> bool:
    """True when the tensor dtype and device can use the FlashAttention kernel."""
    return _get_device_type(q.device) == "cuda" and q.dtype in (torch.float16, torch.bfloat16)


def _build_additive_key_mask(
    bool_mask: torch.Tensor,
    batch: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a boolean padding mask (B, S) -> additive float mask (B, 1, 1, S).

    True  -> token is valid  -> 0.0
    False -> token is pad    -> -inf
    """
    additive = torch.zeros(batch, 1, 1, key_len, device=device, dtype=dtype)
    additive = additive.masked_fill(~bool_mask[:, None, None, :], float("-inf"))
    return additive


class MultiHeadSelfAttention(nn.Module):
    """Batch-first multi-head self-attention.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` which
    uses FlashAttention (CUDA fp16/bf16), memory-efficient attention,
    or a math fallback automatically.  On MPS the fused MPS SDPA path is used.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        causal: bool = False,
        positional_encoding: PositionalEncodingMode = "none",
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive. Got {embedding_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive. Got {num_heads}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")

        positional_encoding = normalize_positional_encoding_mode(positional_encoding)
        if positional_encoding not in ("none", "rope"):
            raise ValueError(
                "Self-attention only supports positional_encoding='none' or 'rope'. "
                f"Got {positional_encoding}."
            )

        if head_dim is None:
            if embedding_dim % num_heads != 0:
                raise ValueError(
                    f"embedding_dim ({embedding_dim}) must be divisible by num_heads "
                    f"({num_heads}) when head_dim is None."
                )
            head_dim = embedding_dim // num_heads

        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be positive. Got {head_dim}. "
                f"Passed {embedding_dim} embedding_dim and {num_heads} num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout
        self.causal = causal
        self.positional_encoding = positional_encoding

        self.qkv_projection = nn.Linear(embedding_dim, 3 * self.inner_dim)
        self.output_projection = nn.Linear(self.inner_dim, embedding_dim)
        self.rope = (
            RotaryEmbedding(head_dim=head_dim, base=rope_base)
            if positional_encoding == "rope"
            else None
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _build_sdpa_mask(
        self,
        mask: torch.Tensor | None,
        tokens: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Combine optional padding mask and causal mask into one additive mask.

        Returns None when no masking is needed (lets SDPA use is_causal fast path).
        """
        additive: torch.Tensor | None = None

        if mask is not None:
            additive = _build_additive_key_mask(mask, batch, tokens, device, dtype)

        if self.causal:
            # Upper-triangular = future positions -> -inf
            causal = torch.full((tokens, tokens), float("-inf"), device=device, dtype=dtype).triu(
                diagonal=1
            )
            additive = causal if additive is None else additive + causal

        return additive

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention to tokens shaped (batch, tokens, dim)."""
        validate_token_tensor(x)

        if x.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}. Got {x.shape[-1]}.")

        if mask is not None:
            validate_mask(mask, x)

        batch, tokens, _ = x.shape
        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._reshape_heads(q)  # (B, H, T, D)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if self.rope is not None:
            cos, sin = self.rope(tokens, x.device, q.dtype)
            q = apply_rope_to_tensor(q, cos, sin)
            k = apply_rope_to_tensor(k, cos, sin)

        # When there is no padding mask and the module is causal we can use the
        # is_causal fast-path (avoids allocating a (T, T) mask on device).
        use_is_causal = self.causal and mask is None
        attn_mask = (
            None if use_is_causal else self._build_sdpa_mask(mask, tokens, batch, x.device, q.dtype)
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.inner_dim)
        return self.output_projection(out)


class MultiHeadCrossAttention(nn.Module):
    """Batch-first multi-head cross-attention.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` which
    dispatches to FlashAttention (CUDA fp16/bf16), memory-efficient attention,
    or a math fallback automatically.  On MPS the fused MPS SDPA path is used.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive. Got {query_dim}.")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive. Got {context_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive. Got {num_heads}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}.")

        if head_dim is None:
            if query_dim % num_heads != 0:
                raise ValueError(
                    f"query_dim ({query_dim}) must be divisible by num_heads "
                    f"({num_heads}) when head_dim is None."
                )
            head_dim = query_dim // num_heads

        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be positive. Got {head_dim}. "
                f"Passed {query_dim} query_dim and {num_heads} num_heads."
            )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.query_projection = nn.Linear(query_dim, self.inner_dim)
        self.key_projection = nn.Linear(context_dim, self.inner_dim)
        self.value_projection = nn.Linear(context_dim, self.inner_dim)
        self.output_projection = nn.Linear(self.inner_dim, query_dim)
        self.dropout = dropout

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-attention from x to context."""
        validate_token_tensor(x)
        validate_token_tensor(context, name="context")

        if x.shape[-1] != self.query_dim:
            raise ValueError(f"Expected query_dim={self.query_dim}. Got {x.shape[-1]}.")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}. Got {context.shape[-1]}.")
        if context.shape[0] != x.shape[0]:
            raise ValueError("context batch size must match x batch size.")

        if context_mask is not None:
            validate_mask(context_mask, context, name="context_mask")

        batch, query_tokens, _ = x.shape
        context_len = context.shape[1]

        q = self._reshape_heads(self.query_projection(x))  # (B, H, Tq, D)
        k = self._reshape_heads(self.key_projection(context))  # (B, H, Tc, D)
        v = self._reshape_heads(self.value_projection(context))  # (B, H, Tc, D)

        attn_mask: torch.Tensor | None = None
        if context_mask is not None:
            attn_mask = _build_additive_key_mask(
                context_mask, batch, context_len, x.device, q.dtype
            )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # cross-attention is never causal
        )

        out = out.transpose(1, 2).contiguous().view(batch, query_tokens, self.inner_dim)
        return self.output_projection(out)
