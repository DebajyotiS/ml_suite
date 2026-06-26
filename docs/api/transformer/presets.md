# Transformer presets

`ml_suite.models.transformer.presets` contains factory functions that assemble fully configured transformer models for common tasks. Each function is a thin wrapper over the core model classes and returns a ready-to-use `nn.Module`.

---

## make_point_cloud_classifier

No positional encoding is applied because point clouds are unordered and predictions should be permutation-invariant.

::: ml_suite.models.transformer.presets.make_point_cloud_classifier

---

## make_point_to_point_model

No positional encoding is applied because point clouds are unordered and per-point outputs should be permutation-equivariant.

::: ml_suite.models.transformer.presets.make_point_to_point_model

---

## make_conditioned_point_to_point_model

No positional encoding is applied because point clouds are unordered and per-point outputs should be permutation-equivariant.

::: ml_suite.models.transformer.presets.make_conditioned_point_to_point_model

---

## make_sequence_classifier

RoPE is used here because it encodes relative positions inside attention rather than as an additive vector, which generalizes better to variable-length sequences.

::: ml_suite.models.transformer.presets.make_sequence_classifier

---

## make_patch_grid_model

Sinusoidal encoding is used because patch grids have a fixed spatial structure and sinusoidal embeddings generalize to unseen grid sizes without retraining.

::: ml_suite.models.transformer.presets.make_patch_grid_model

---

## make_patch_classifier

Sinusoidal encoding is used because patch grids have a fixed spatial structure and sinusoidal embeddings generalize to unseen grid sizes without retraining.

::: ml_suite.models.transformer.presets.make_patch_classifier
