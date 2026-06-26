# Linear models

`ml_suite.models.linear` provides linear projection blocks and multi-layer perceptrons with optional context conditioning via concatenation, addition, multiplication, FiLM, or cross-attention.

---

## Linear blocks

`LinearBlock` is the basic unit for MLP-style networks in this library. It wraps a single linear projection and optionally accepts a secondary conditioning tensor. The `context_injection` parameter controls how that second signal is mixed in. For most conditioning tasks, `film` is the right choice because it applies a learned scale and shift per feature, which is what diffusion models use to inject time information. Use `concat` when you want the simplest possible thing and are happy for the network to learn the mixing from scratch.

**Choosing `context_injection`:** `concat` appends context to the input before projection, giving the network full control over the mixing. `add` and `multiply` are element-wise and useful when context and input are already in the same space. `film` applies learned scale and shift and is the standard choice for diffusion timestep conditioning. `cross_attn` applies multi-head attention and requires context to have shape `(batch, tokens, dim)` rather than `(batch, dim)`.

::: ml_suite.models.linear.blocks

---

## VAD inference

!!! warning "Not yet implemented"
    The classes in this section are experimental placeholders. `VADLinearBlock.forward` raises `NotImplementedError`. The API may change before implementation is complete.

::: ml_suite.models.linear.vad
