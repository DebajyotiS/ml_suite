# Transformer primitives

These are the low-level components that the model classes in `models.transformer` are assembled from. You only need to read this page if you are building a non-standard architecture by hand. For everything else, start from the presets or models pages.

---

## Attention

::: ml_suite.models.transformer.attention

---

## Blocks and stack

::: ml_suite.models.transformer.blocks

::: ml_suite.models.transformer.stacks

---

## Tokenizers

These tokenizers convert raw inputs into the `(batch, tokens, embedding_dim)` format the transformer expects. Use `ContinuousInputTokenizer` for float feature vectors. Use `PatchTokenizerND` for spatial grids. Use `DiscreteTokenTokenizer` for integer token IDs. `SetTokenizer` is identical to `ContinuousInputTokenizer` and exists as a semantic alias for unordered inputs.

::: ml_suite.models.transformer.tokenization

---

## Positional encodings

Use `SinusoidalPositionalEmbedding` for patch transformers and other tasks where you want a fixed encoding that generalizes to unseen sequence lengths. Use `LearnedPositionalEmbedding` when sequence lengths are fixed and the positions can be learned. Use `RotaryEmbedding` via the `positional_encoding='rope'` argument to self-attention when you want position-relative attention without an additive embedding, which is standard for sequence models. Pass `positional_encoding='none'` for point clouds and other inputs where order does not matter.

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
