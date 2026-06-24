"""Linear transformation primitives and MLP stacks."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from ml_suite.utils.activations import get_activation


class LinearBlock(nn.Module):
    """A linear transformation primitive supporting context conditioning and residual routing.

    This block wraps a standard linear mapping while offering multiple ways to inject secondary
    context tensors (e.g., conditioning tokens, embeddings, or timesteps). The layer ordering
    executes as follows: Context Conditioning -> Linear Projection -> Residual Addition -> Activation.

    Args:
        input_dim: Number of features in the input tensor.
        output_dim: Number of features produced by the main linear projection.
        context_dim: Number of features in the conditioning context tensor. Set to 0 to disable.
        context_injection: The mechanism used to merge the context tensor into the main stream:
            - 'concat': Concatenates input and context along the feature dimension before projection.
            - 'add': Projects context to match input features and sums them.
            - 'multiply': Projects context to match input features and multiplies them.
            - 'film': Computes feature-wise affine scaling and shifting parameters from the context.
            - 'cross_attn': Applies a multi-head cross-attention mechanism over the context.
        activation: Name of the activation function to apply.
        do_residual: If True, adds the input tensor to the linear projection output.

    Raises:
        ValueError: If `do_residual` is True but `input_dim` does not equal `output_dim`.
        ValueError: If an unsupported `context_injection` strategy is provided.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: int = 0,
        context_injection: Literal["concat", "add", "multiply", "film", "cross_attn"] = "concat",
        activation: str = "silu",
        do_residual: bool = False,
    ) -> None:
        super().__init__()
        if do_residual and input_dim != output_dim:
            raise ValueError(
                f"Residual connection requires input_dim ({input_dim}) to equal output_dim ({output_dim})."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.has_context = context_dim > 0
        self.context_injection = context_injection
        self.do_residual = do_residual
        self.activation = activation

        net = nn.Linear(input_dim, output_dim)

        if self.has_context:
            match context_injection:
                case "concat":
                    net = nn.Linear(input_dim + context_dim, output_dim)
                case "add":
                    self.context_proj = nn.Linear(context_dim, input_dim)
                case "multiply":
                    self.context_proj = nn.Linear(context_dim, input_dim)
                case "film":
                    self.film_scale = nn.Linear(context_dim, input_dim)
                    self.film_shift = nn.Linear(context_dim, input_dim)
                case "cross_attn":
                    self.context_proj = nn.Linear(context_dim, input_dim)
                    self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
                case _:
                    raise ValueError(f"Unsupported context injection method: {context_injection}")

        self.net = net
        self.activation_fn = get_activation(activation)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.has_context and context is None:
            raise ValueError("Context tensor is required but not provided.")

        if self.has_context and context is not None:
            match self.context_injection:
                case "concat":
                    x = torch.cat([x, context], dim=-1)
                case "add":
                    x = x + self.context_proj(context)
                case "multiply":
                    x = x * self.context_proj(context)
                case "film":
                    scale = self.film_scale(context)
                    shift = self.film_shift(context)
                    x = x * (scale + 1) + shift
                case "cross_attn":
                    projected_context = self.context_proj(context)
                    x_shape = x.shape
                    x_flat = x.view(-1, x_shape[-1]).unsqueeze(0)
                    context_flat = projected_context.view(
                        -1, projected_context.shape[-1]
                    ).unsqueeze(0)
                    attn_output, _ = self.cross_attn(x_flat, context_flat, context_flat)
                    x = attn_output.squeeze(0).view(x_shape)

        out = self.net(x)
        if self.do_residual:
            out = out + x
        return self.activation_fn(out)

    def __repr__(self) -> str:
        return (
            f"LinearBlock(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"context_dim={self.context_dim}, context_injection='{self.context_injection}', "
            f"activation='{self.activation}', do_residual={self.do_residual})"
        )

    def __str__(self) -> str:
        residual_str = " + x" if self.do_residual else ""

        if self.has_context:
            if self.context_injection == "concat":
                expr = "Linear(concat(x, context))"
            elif self.context_injection == "film":
                expr = "Linear(x * (scale + 1) + shift)"
            elif self.context_injection == "cross_attn":
                expr = "Linear(CrossAttention(x, context))"
            else:
                expr = f"Linear(x {'+' if self.context_injection == 'add' else '*'} proj(context))"
        else:
            expr = "Linear(x)"

        return f"{self.activation_fn.__class__.__name__}({expr}{residual_str})"


class MLP(nn.Module):
    """A Multi-Layer Perceptron network composed of sequential LinearBlock primitives.

    Args:
        input_dim: Feature dimension entering the first network layer.
        hidden_dim: Default feature dimension produced by the final output layer block.
        num_layers: Total number of LinearBlock layers to create. Must be at least 1.
        context_dim: Feature dimension of the conditioning tensor. Set to 0 to disable.
        context_injection: The mechanism used to merge context vectors inside each block.
        hidden_layers: Sequence of custom feature hidden dimensions between layers.
            Must contain exactly `num_layers - 1` elements. If None, defaults to `hidden_dim`.
        activation: Default activation name applied across all layer blocks.
        activation_list: Explicit list of activation names for each layer.
            Must contain exactly `num_layers` elements.
        do_residual: If True, blocks with matching input and output sizes use residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        context_dim: int = 0,
        context_injection: Literal["concat", "add", "multiply", "film", "cross_attn"] = "concat",
        hidden_layers: List[int] | None = None,
        activation: str = "silu",
        activation_list: List[str] | None = None,
        do_residual: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.context_injection = context_injection
        self.do_residual = do_residual

        layers = [hidden_dim] * (num_layers - 1) if hidden_layers is None else hidden_layers
        activations = [activation] * num_layers if activation_list is None else activation_list

        if len(layers) != num_layers - 1:
            raise ValueError(
                f"Length of hidden_layers ({len(layers)}) must be num_layers - 1 ({num_layers - 1})."
            )
        if len(activations) != num_layers:
            raise ValueError(
                f"Length of activation_list ({len(activations)}) must be num_layers ({num_layers})."
            )

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else layers[i - 1]
            out_dim = layers[i] if i < num_layers - 1 else hidden_dim
            block_residual = do_residual and (in_dim == out_dim)
            block = LinearBlock(
                input_dim=in_dim,
                output_dim=out_dim,
                context_dim=context_dim,
                context_injection=context_injection,
                activation=activations[i],
                do_residual=block_residual,
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, context)
        return x

    def __repr__(self) -> str:
        return (
            f"MLP(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={len(self.blocks)}, context_dim={self.context_dim}, "
            f"context_injection='{self.context_injection}', do_residual={self.do_residual})"
        )

    def __str__(self) -> str:
        context_str = f"Injected via '{self.context_injection}'" if self.context_dim > 0 else "None"
        lines = [
            f"MLP (input_dim={self.input_dim} -> hidden_dim={self.hidden_dim})",
            f"  ├── Context: dim={self.context_dim}, Type={context_str}",
            f"  └── Blocks Stack:",
        ]
        for i, block in enumerate(self.blocks):
            lines.append(f"        [{i}] └── {block}")
        return "\n".join(lines)
