"""Token-centric transformer models.

All core models operate on token tensors shaped:

    tokens.shape == (batch, tokens, embedding_dim)

Modalities enter through tokenizers and leave through heads or decoders.
"""

from .types import (
    AttentionType,
    NormType,
    PatchOutputMode,
    PoolingMode,
    PositionalEncodingMode,
    TimeEmbeddingType,
)
from .tokenization import (
    ContinuousInputTokenizer,
    DiscreteTokenTokenizer,
    PatchTokenizerND,
    SetTokenizer,
)
from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from .blocks import FeedForward, TransformerBlock
from .stacks import TransformerStack
from .conditioning import (
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    TransformerConditioningBuilder,
    ConditionTokenProjector,
)
from .pooling import TokenPooling
from .heads import (
    TokenwiseHead,
    PooledHead,
    ClassificationHead,
    RegressionHead,
)
from .decoders import TokenDecoder, PatchDecoderND, QuerySetDecoder
from .models import (
    TokenToTokenTransformer,
    TokenToVectorTransformer,
    TokenToClassTransformer,
    ConditionedTokenTransformer,
    PatchTransformerND,
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
