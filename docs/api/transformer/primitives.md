# Transformer primitives

Lower-level building blocks used by the transformer models: attention layers, feed-forward blocks, the transformer stack, tokenizers, positional encodings, heads, decoders, and conditioning utilities.

---

## Attention

### MultiHeadSelfAttention

::: ml_suite.models.transformer.attention.MultiHeadSelfAttention

### MultiHeadCrossAttention

::: ml_suite.models.transformer.attention.MultiHeadCrossAttention

---

## Blocks and stack

### FeedForward

::: ml_suite.models.transformer.blocks.FeedForward

### TransformerBlock

::: ml_suite.models.transformer.blocks.TransformerBlock

### TransformerStack

::: ml_suite.models.transformer.stacks.TransformerStack

---

## Tokenizers

### ContinuousInputTokenizer

::: ml_suite.models.transformer.tokenization.ContinuousInputTokenizer

### DiscreteTokenTokenizer

::: ml_suite.models.transformer.tokenization.DiscreteTokenTokenizer

### PatchTokenizerND

::: ml_suite.models.transformer.tokenization.PatchTokenizerND

### SetTokenizer

::: ml_suite.models.transformer.tokenization.SetTokenizer

---

## Positional encodings

### LearnedPositionalEmbedding

::: ml_suite.models.transformer.positional.LearnedPositionalEmbedding

### SinusoidalPositionalEmbedding

::: ml_suite.models.transformer.positional.SinusoidalPositionalEmbedding

### RotaryEmbedding

::: ml_suite.models.transformer.positional.RotaryEmbedding

---

## Heads

### TokenwiseHead

::: ml_suite.models.transformer.heads.TokenwiseHead

### PooledHead

::: ml_suite.models.transformer.heads.PooledHead

### ClassificationHead

::: ml_suite.models.transformer.heads.ClassificationHead

### RegressionHead

::: ml_suite.models.transformer.heads.RegressionHead

---

## Decoders

### TokenDecoder

::: ml_suite.models.transformer.decoders.TokenDecoder

### PatchDecoderND

::: ml_suite.models.transformer.decoders.PatchDecoderND

### QuerySetDecoder

::: ml_suite.models.transformer.decoders.QuerySetDecoder

---

## Conditioning

### TransformerConditioningBuilder

::: ml_suite.models.transformer.conditioning.TransformerConditioningBuilder

### ConditionTokenProjector

::: ml_suite.models.transformer.conditioning.ConditionTokenProjector
