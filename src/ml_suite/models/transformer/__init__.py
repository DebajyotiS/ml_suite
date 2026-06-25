"""Token-centric transformer models.

All core models operate on token tensors shaped:

    tokens.shape == (batch, tokens, embedding_dim)

Modalities enter through tokenizers and leave through heads or decoders.
"""

from .attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from .blocks import FeedForward, TransformerBlock
from .conditioning import (
    ConditionTokenProjector,
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    TransformerConditioningBuilder,
)
from .decoders import PatchDecoderND, QuerySetDecoder, TokenDecoder
from .heads import (
    ClassificationHead,
    PooledHead,
    RegressionHead,
    TokenwiseHead,
)
from .models import (
    ConditionedTokenTransformer,
    PatchTransformerND,
    TokenToClassTransformer,
    TokenToTokenTransformer,
    TokenToVectorTransformer,
)
from .pooling import TokenPooling
from .stacks import TransformerStack
from .tokenization import (
    ContinuousInputTokenizer,
    DiscreteTokenTokenizer,
    PatchTokenizerND,
    SetTokenizer,
)
from .types import (
    AttentionType,
    NormType,
    PatchOutputMode,
    PoolingMode,
    PositionalEncodingMode,
    TimeEmbeddingType,
)

__all__ = [
    "AttentionType",
    "NormType",
    "PatchOutputMode",
    "PoolingMode",
    "PositionalEncodingMode",
    "TimeEmbeddingType",
    "ContinuousInputTokenizer",
    "DiscreteTokenTokenizer",
    "PatchTokenizerND",
    "SetTokenizer",
    "MultiHeadSelfAttention",
    "MultiHeadCrossAttention",
    "FeedForward",
    "TransformerBlock",
    "TransformerStack",
    "SinusoidalTimeEmbedding",
    "TimeEmbeddingMLP",
    "TransformerConditioningBuilder",
    "ConditionTokenProjector",
    "TokenPooling",
    "TokenwiseHead",
    "PooledHead",
    "ClassificationHead",
    "RegressionHead",
    "TokenDecoder",
    "PatchDecoderND",
    "QuerySetDecoder",
    "TokenToTokenTransformer",
    "TokenToVectorTransformer",
    "TokenToClassTransformer",
    "ConditionedTokenTransformer",
    "PatchTransformerND",
]
