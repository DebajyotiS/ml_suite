# Transformer models

`ml_suite.models.transformer` provides token-centric transformer architectures. All models operate on token tensors shaped `(batch, tokens, dim)` and enter and exit via tokenizers and heads.

This module contains five model classes. `TokenToTokenTransformer` maps a sequence to a sequence of the same length. `TokenToVectorTransformer` collapses the sequence to a single vector via pooling. `TokenToClassTransformer` is the same but the output is a logit vector. `ConditionedTokenTransformer` is `TokenToTokenTransformer` with time, class, and context conditioning. `PatchTransformerND` handles spatial grid inputs by splitting them into patches first. For common configurations, the preset functions in `models.transformer.presets` are easier to use than constructing these directly.

---

## TokenToTokenTransformer

Use this when you want each input token to produce one output token, for example flow-matching velocity estimation on point sets.

::: ml_suite.models.transformer.models.TokenToTokenTransformer

---

## TokenToVectorTransformer

Use this when you want the whole sequence collapsed to one vector, for example encoding a variable-length sequence for downstream regression.

::: ml_suite.models.transformer.models.TokenToVectorTransformer

---

## TokenToClassTransformer

Use this for classification tasks. It is a thin wrapper over `TokenToVectorTransformer` that names the output dimension `num_classes`.

::: ml_suite.models.transformer.models.TokenToClassTransformer

---

## ConditionedTokenTransformer

Use this for score or velocity field estimation in diffusion or flow-matching models operating on token sequences. Conditioning signals are summed and broadcast-added to every token before the transformer stack.

Note that conditioning is global, not per-token. All global signals (`time`, `class_labels`, `global_context`) are projected to `embedding_dim`, summed, and added to every token in the sequence. If you need per-token conditioning, use cross-attention via `cross_attention_dim` instead.

::: ml_suite.models.transformer.models.ConditionedTokenTransformer

---

## PatchTransformerND

Use this for grid-structured inputs where you want the transformer to operate on spatial patches rather than individual spatial positions. The `output_mode` parameter controls whether you get back a grid, a token sequence, or a single pooled vector.

::: ml_suite.models.transformer.models.PatchTransformerND
