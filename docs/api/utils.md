# Utilities

`ml_suite.utils` provides shared utilities used across model families. You do not need to use these directly in most cases because the `Conditioned*` models wire them up internally. They are exposed here for custom architectures that need to build conditioning pipelines by hand.

---

## Activations

`get_activation` is a factory that returns an `nn.Module` for a named activation function. Pass the name as a string anywhere in the library that accepts an `activation` argument. The factory exists so you can configure activation choice via a config file or command-line flag without importing the activation class directly.

::: ml_suite.utils.activations

---

## Time embeddings

These classes convert a scalar timestep `t` (shape `(batch,)`) into a vector that can be used as a conditioning signal. `SinusoidalTimeEmbedding` produces a fixed sinusoidal encoding and is a good default. `TimeEmbeddingMLP` follows it with a learned projection, which gives the network more capacity to learn how to use the time signal. In practice, `TimeEmbeddingMLP` is what most diffusion and flow-matching models use.

::: ml_suite.utils.conditioning
