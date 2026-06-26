# U-Net models

`ml_suite.models.unet` provides encoder-decoder U-Nets for 1D, 2D, and 3D data with skip connections, optional self- and cross-attention, and a rich conditioning interface (time, class, global context).

Most users will only need `UNet` or `ConditionedUNet`. Use `UNet` for tasks where the model takes only the data tensor as input, like segmentation. Use `ConditionedUNet` when the model needs additional signals at every layer, such as a diffusion timestep, a class label, or a global context vector. The internal stage, attention, and conditioning classes documented below are building blocks for custom architectures.

---

## Models

### UNet

::: ml_suite.models.unet.models.UNet

---

### ConditionedUNet

All global conditioning sources, meaning `time`, `class_labels`, and `global_context`, are each projected to `condition_dim` and then summed into a single vector. That vector is passed to every convolutional block as a FiLM signal. The FiLM projection is zero-initialized, so at the start of training the model behaves identically to an unconditioned UNet and the conditioning signal is learned gradually. Token-level conditioning via `cross_context` goes only to the spatial attention blocks.

::: ml_suite.models.unet.models.ConditionedUNet

**Multi-signal forward examples:**

```python
# Time-conditioned with class labels
out = model(x, time=t, class_labels=labels)

# With cross-attention context (e.g. text conditioning)
out = model(x, time=t, cross_context=text_embeddings, cross_context_mask=mask)

# All signals at once
out = model(x, time=t, class_labels=labels,
            global_context=ctx, cross_context=text_embeddings)
```

---

## Internal components

??? note "Stages"

    ::: ml_suite.models.unet.stages

??? note "Attention"

    ::: ml_suite.models.unet.attention

??? note "Conditioning"

    ::: ml_suite.models.unet.conditioning
        options:
          members: [ConditioningBuilder]
