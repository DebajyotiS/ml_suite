# Transformer primitives

Lower-level building blocks used by the transformer models: attention layers, feed-forward blocks, the transformer stack, tokenizers, positional encodings, heads, decoders, and conditioning utilities.

---

## Attention

::: ml_suite.models.transformer.attention

---

## Blocks and stack

::: ml_suite.models.transformer.blocks

::: ml_suite.models.transformer.stacks

---

## Tokenizers

> **Note on naming:** "Tokenizers" here means the components that convert raw modality inputs into the `(B, T, embedding_dim)` format the transformer expects — including input projection for continuous features, embedding lookup for discrete IDs, and patch extraction for spatial data. Upstream segmentation (e.g. BPE) is handled outside these classes and is out of scope.

::: ml_suite.models.transformer.tokenization

---

## Positional encodings

::: ml_suite.models.transformer.positional

---

## Heads

::: ml_suite.models.transformer.heads

---

## Decoders

::: ml_suite.models.transformer.decoders

---

## Conditioning

::: ml_suite.models.transformer.conditioning
    options:
      members: [TransformerConditioningBuilder, ConditionTokenProjector]
