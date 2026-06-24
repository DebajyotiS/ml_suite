"""Dimension-agnostic convolutional primitives and ConvNet backbones."""

from typing import Literal, Sequence

import torch
from torch import nn

from ml_suite.utils.activations import get_activation


NormType = Literal["batch", "group", "layer"] | None
DownsampleMode = Literal["stride", "pool"]
GlobalPoolMode = Literal["avg", "max", "cat_avg_max"]


class ConvBlock(nn.Module):
    """A dimension-agnostic convolutional block.

    Operation order:
        Conv -> Norm -> Residual Add -> Activation
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_dim: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "silu",
        norm_type: NormType = "batch",
        num_groups: int = 32,
        do_residual: bool = False,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive. Got {input_channels}.")
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive. Got {output_channels}.")
        if conv_dim not in (1, 2, 3):
            raise ValueError(f"Unsupported conv_dim: {conv_dim}. Choose from 1, 2, or 3.")
        if do_residual and input_channels != output_channels:
            raise ValueError(
                "Residual connection requires input_channels to equal output_channels."
            )

        conv_mappings = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        self.conv_dim = conv_dim
        self.conv = conv_mappings[conv_dim](
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = self._build_norm(conv_dim, output_channels, norm_type, num_groups)
        self.activation_fn = get_activation(activation)
        self.do_residual = do_residual

    @staticmethod
    def _build_norm(
        conv_dim: int,
        output_channels: int,
        norm_type: NormType,
        num_groups: int,
    ) -> nn.Module | None:
        if norm_type == "batch":
            return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[conv_dim](
                output_channels
            )
        if norm_type == "group":
            if num_groups <= 0:
                raise ValueError(f"num_groups must be positive. Got {num_groups}.")
            if output_channels % num_groups != 0:
                raise ValueError(
                    f"output_channels ({output_channels}) must be divisible by "
                    f"num_groups ({num_groups})."
                )
            return nn.GroupNorm(num_groups=num_groups, num_channels=output_channels)
        if norm_type == "layer":
            # GroupNorm with one group is LayerNorm-like for convolutional tensors.
            return nn.GroupNorm(num_groups=1, num_channels=output_channels)
        if norm_type is None:
            return None
        raise ValueError(
            f"Unsupported norm_type: {norm_type}. Choose 'batch', 'group', 'layer', or None."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.do_residual:
            if out.shape != x.shape:
                raise ValueError(
                    f"Residual connection shape mismatch: input {x.shape} vs output {out.shape}."
                )
            out = x + out
        return self.activation_fn(out)

    def __repr__(self) -> str:
        norm_name = self.norm.__class__.__name__ if self.norm else "None"
        return (
            f"ConvBlock(input_channels={self.conv.in_channels}, "
            f"output_channels={self.conv.out_channels}, "
            f"conv_dim={self.conv_dim}, "
            f"kernel_size={self.conv.kernel_size}, "
            f"stride={self.conv.stride}, "
            f"padding={self.conv.padding}, "
            f"norm={norm_name}, "
            f"activation={self.activation_fn.__class__.__name__}, "
            f"do_residual={self.do_residual})"
        )

    def __str__(self) -> str:
        expr = (
            f"Conv{self.conv_dim}D({self.conv.in_channels}->{self.conv.out_channels}, "
            f"k={self.conv.kernel_size}, s={self.conv.stride})"
        )
        if self.norm is not None:
            expr = f"{self.norm.__class__.__name__}({expr})"
        if self.do_residual:
            expr = f"x + {expr}"
        return f"{self.activation_fn.__class__.__name__}({expr})"


class ConditionedConvBlock(ConvBlock):
    """A ConvBlock with FiLM conditioning.

    The context vector has shape ``(batch, context_dim)`` and is projected into
    per-channel scale and shift parameters. The FiLM projection is zero-initialized,
    so the block starts as the matching unconditioned ConvBlock.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        context_dim: int,
        **kwargs,
    ) -> None:
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive. Got {context_dim}.")
        super().__init__(input_channels=input_channels, output_channels=output_channels, **kwargs)
        self.context_dim = context_dim
        self.context_projection = nn.Linear(context_dim, 2 * output_channels)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch, context_dim). Got {context.shape}.")
        if context.shape[0] != x.shape[0]:
            raise ValueError(
                f"context batch size ({context.shape[0]}) must match input batch size "
                f"({x.shape[0]})."
            )
        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"context feature dimension must be {self.context_dim}. Got {context.shape[1]}."
            )

        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)

        scale, shift = self.context_projection(context).chunk(2, dim=1)
        view_shape = (x.shape[0], self.conv.out_channels) + (1,) * len(x.shape[2:])
        out = out * (1.0 + scale.view(view_shape)) + shift.view(view_shape)

        if self.do_residual:
            if out.shape != x.shape:
                raise ValueError(
                    f"Residual connection shape mismatch: input {x.shape} vs output {out.shape}."
                )
            out = x + out
        return self.activation_fn(out)


class ConvNet(nn.Module):
    """A flexible 1D, 2D, or 3D convolutional backbone or classifier."""

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        stage_channels: Sequence[int],
        blocks_per_stage: int | Sequence[int],
        downsample_mode: DownsampleMode = "stride",
        norm_type: NormType = "batch",
        global_pool_mode: GlobalPoolMode = "avg",
        num_classes: int | None = None,
        activation: str = "silu",
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            conv_dim, in_channels, stage_channels, downsample_mode, global_pool_mode, num_classes
        )
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.stage_channels = list(stage_channels)
        self.downsample_mode = downsample_mode
        self.norm_type = norm_type
        self.global_pool_mode = global_pool_mode
        self.num_classes = num_classes
        self.activation = activation
        self.num_groups = num_groups
        self.blocks_per_stage = self._resolve_blocks_per_stage(
            blocks_per_stage, len(self.stage_channels)
        )

        self.stem = ConvBlock(
            in_channels,
            self.stage_channels[0],
            conv_dim=conv_dim,
            activation=activation,
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.stages = self._build_stages()
        self.head = self._build_head(self.stage_channels[-1], num_classes)

    @staticmethod
    def _validate_init_args(
        conv_dim: int,
        in_channels: int,
        stage_channels: Sequence[int],
        downsample_mode: str,
        global_pool_mode: str,
        num_classes: int | None,
    ) -> None:
        if conv_dim not in (1, 2, 3):
            raise ValueError(f"conv_dim must be 1, 2, or 3. Got {conv_dim}.")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive. Got {in_channels}.")
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must contain at least one stage definition.")
        if any(ch <= 0 for ch in stage_channels):
            raise ValueError(f"All stage channels must be positive. Got {stage_channels}.")
        if downsample_mode not in ("stride", "pool"):
            raise ValueError(f"downsample_mode must be 'stride' or 'pool'. Got {downsample_mode}.")
        if global_pool_mode not in ("avg", "max", "cat_avg_max"):
            raise ValueError(
                f"global_pool_mode must be 'avg', 'max', or 'cat_avg_max'. Got {global_pool_mode}."
            )
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"num_classes must be positive or None. Got {num_classes}.")

    @staticmethod
    def _resolve_blocks_per_stage(
        blocks_per_stage: int | Sequence[int],
        num_stages: int,
    ) -> list[int]:
        if isinstance(blocks_per_stage, int):
            resolved = [blocks_per_stage] * num_stages
        else:
            if len(blocks_per_stage) != num_stages:
                raise ValueError(
                    f"Length of blocks_per_stage ({len(blocks_per_stage)}) "
                    f"must match number of stages ({num_stages})."
                )
            resolved = list(blocks_per_stage)
        if any(n < 1 for n in resolved):
            raise ValueError("Every stage must contain at least one block.")
        return resolved

    def _make_conv_block(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        do_residual: bool,
    ) -> nn.Module:
        return ConvBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            conv_dim=self.conv_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=self.activation,
            norm_type=self.norm_type,
            num_groups=self.num_groups,
            do_residual=do_residual,
        )

    def _make_pool(self) -> nn.Module:
        return {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[self.conv_dim](
            kernel_size=2, stride=2
        )

    def _build_stages(self) -> nn.ModuleList:
        stages = nn.ModuleList()
        current_channels = self.stage_channels[0]
        for stage_idx, target_channels in enumerate(self.stage_channels):
            stage_modules = nn.ModuleList()
            for block_idx in range(self.blocks_per_stage[stage_idx]):
                is_transition = stage_idx > 0 and block_idx == 0
                if is_transition and self.downsample_mode == "stride":
                    stage_modules.append(
                        self._make_conv_block(current_channels, target_channels, 2, False)
                    )
                elif is_transition and self.downsample_mode == "pool":
                    stage_modules.append(
                        self._make_conv_block(current_channels, target_channels, 1, False)
                    )
                    stage_modules.append(self._make_pool())
                else:
                    block_in = target_channels if block_idx > 0 else current_channels
                    stage_modules.append(
                        self._make_conv_block(
                            block_in, target_channels, 1, block_in == target_channels
                        )
                    )
            stages.append(stage_modules)
            current_channels = target_channels
        return stages

    def _build_head(self, final_channels: int, num_classes: int | None) -> nn.Module | None:
        if num_classes is None:
            self.global_avg_pool = None
            self.global_max_pool = None
            return None
        self.global_avg_pool = {
            1: nn.AdaptiveAvgPool1d,
            2: nn.AdaptiveAvgPool2d,
            3: nn.AdaptiveAvgPool3d,
        }[self.conv_dim](1)
        self.global_max_pool = {
            1: nn.AdaptiveMaxPool1d,
            2: nn.AdaptiveMaxPool2d,
            3: nn.AdaptiveMaxPool3d,
        }[self.conv_dim](1)
        in_features = (
            final_channels * 2 if self.global_pool_mode == "cat_avg_max" else final_channels
        )
        return nn.Linear(in_features, num_classes)

    def _apply_head(self, out: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            return out
        if self.global_pool_mode == "avg":
            out = self.global_avg_pool(out)
        elif self.global_pool_mode == "max":
            out = self.global_max_pool(out)
        else:
            out = torch.cat([self.global_avg_pool(out), self.global_max_pool(out)], dim=1)
        return self.head(torch.flatten(out, start_dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            for module in stage:
                out = module(out)
        return self._apply_head(out)

    def print_receptive_field(self) -> None:
        """Print the analytical receptive field after each convolution or pool.

        Tracks cumulative receptive field and cumulative stride across the
        convolutional backbone. This is a diagnostic utility only.
        """
        current_rf = 1.0
        current_stride = 1.0

        print(
            f"{'Layer / Module':<45} | "
            f"{'Layer RF':<10} | "
            f"{'Cumulative RF':<15} | "
            f"{'Cumulative Stride':<15}"
        )
        print("-" * 95)

        def process_block(name: str, module: nn.Module) -> None:
            nonlocal current_rf, current_stride

            if isinstance(module, ConvBlock):
                kernel_size = module.conv.kernel_size[0]
                stride = module.conv.stride[0]

            elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                kernel_size = (
                    module.kernel_size
                    if isinstance(module.kernel_size, int)
                    else module.kernel_size[0]
                )
                stride = module.stride if isinstance(module.stride, int) else module.stride[0]

            else:
                return

            current_rf = current_rf + (kernel_size - 1) * current_stride
            current_stride = current_stride * stride

            print(f"{name:<45} | {kernel_size:<10} | {current_rf:<15.1f} | {current_stride:<15.1f}")

        process_block("Stem", self.stem)

        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                full_name = f"Stage {stage_idx + 1} - Block {block_idx}"
                process_block(full_name, block)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(conv_dim={self.conv_dim}, "
            f"in_channels={self.in_channels}, "
            f"stage_channels={self.stage_channels}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"downsample_mode='{self.downsample_mode}', "
            f"norm_type='{self.norm_type}', "
            f"global_pool_mode='{self.global_pool_mode}', "
            f"num_classes={self.num_classes}, "
            f"activation='{self.activation}')"
        )

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__} {self.conv_dim}D (in_channels={self.in_channels})"]
        lines.append(f"  ├── Stem: {self.stem}")

        for stage_idx, stage in enumerate(self.stages):
            lines.append(
                f"  ├── Stage {stage_idx + 1} ({self.blocks_per_stage[stage_idx]} blocks):"
            )
            for module in stage:
                lines.append(f"  │     └── {module}")

        if self.head is not None:
            lines.append(f"  ├── Global Pooling: {self.global_pool_mode.upper()}")
            lines.append(f"  └── Head: Linear({self.head.in_features}->{self.head.out_features})")
        else:
            lines.append("  └── Head: None (feature extractor)")

        return "\n".join(lines)


class ConditionedConvNet(ConvNet):
    """FiLM-conditioned ConvNet using ConditionedConvBlock in all stages."""

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        stage_channels: Sequence[int],
        blocks_per_stage: int | Sequence[int],
        context_dim: int,
        downsample_mode: DownsampleMode = "stride",
        norm_type: NormType = "batch",
        global_pool_mode: GlobalPoolMode = "avg",
        num_classes: int | None = None,
        activation: str = "silu",
        num_groups: int = 32,
    ) -> None:
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive. Got {context_dim}.")
        self.context_dim = context_dim
        super().__init__(
            conv_dim=conv_dim,
            in_channels=in_channels,
            stage_channels=stage_channels,
            blocks_per_stage=blocks_per_stage,
            downsample_mode=downsample_mode,
            norm_type=norm_type,
            global_pool_mode=global_pool_mode,
            num_classes=num_classes,
            activation=activation,
            num_groups=num_groups,
        )

    def _make_conv_block(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        do_residual: bool,
    ) -> nn.Module:
        return ConditionedConvBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            context_dim=self.context_dim,
            conv_dim=self.conv_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=self.activation,
            norm_type=self.norm_type,
            num_groups=self.num_groups,
            do_residual=do_residual,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            for module in stage:
                if isinstance(module, ConditionedConvBlock):
                    out = module(out, context)
                else:
                    out = module(out)
        return self._apply_head(out)

    def __repr__(self) -> str:
        return (
            f"ConditionedConvNet(context_dim={self.context_dim}, "
            f"conv_dim={self.conv_dim}, "
            f"in_channels={self.in_channels}, "
            f"stage_channels={self.stage_channels}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"downsample_mode='{self.downsample_mode}', "
            f"norm_type='{self.norm_type}', "
            f"global_pool_mode='{self.global_pool_mode}', "
            f"num_classes={self.num_classes}, "
            f"activation='{self.activation}')"
        )
