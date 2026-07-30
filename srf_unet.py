"""SRF-UNet model definitions for lightweight retinal vessel segmentation."""

from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = [
    "MODEL_CHANNELS",
    "BipolarCrossInteractionFusion",
    "CoordinateAttention",
    "SRFDepthwiseConvolution",
    "SRFABlock",
    "SRFUNet",
    "SRF_UNet",
    "create_model",
    "srf_unet_s",
    "srf_unet",
    "srf_unet_l",
]


MODEL_CHANNELS: Dict[str, Tuple[int, int, int, int, int]] = {
    "s": (8, 16, 32, 48, 80),
    "base": (16, 32, 64, 96, 160),
    "l": (32, 64, 128, 192, 320),
}

_SIZE_ALIASES = {
    "s": "s",
    "small": "s",
    "base": "base",
    "b": "base",
    "l": "l",
    "large": "l",
}


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _activation(name: str, inplace: bool = True) -> nn.Module:
    activations = {
        "relu": lambda: nn.ReLU(inplace=inplace),
        "relu6": lambda: nn.ReLU6(inplace=inplace),
        "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=inplace),
        "gelu": nn.GELU,
        "hard_swish": lambda: nn.Hardswish(inplace=inplace),
    }
    key = name.lower()
    if key not in activations:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation '{name}'. Choose from: {choices}.")
    return activations[key]()


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch, channels, height, width = x.shape
    if groups <= 0 or channels % groups:
        raise ValueError(
            f"Channel shuffle requires channels divisible by groups, got "
            f"channels={channels}, groups={groups}."
        )
    channels_per_group = channels // groups
    x = x.reshape(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(batch, channels, height, width)


class CoordinateAttention(nn.Module):
    """Preserve height- and width-aware positional responses."""

    def __init__(
        self,
        channels: int,
        reduction: int = 32,
        min_channels: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")

        hidden_channels = max(min_channels, channels // reduction)
        self.shared_projection = nn.Conv2d(
            channels, hidden_channels, kernel_size=1, bias=False
        )
        self.normalization = nn.BatchNorm2d(hidden_channels)
        self.activation = nn.Hardswish(inplace=True)
        self.height_projection = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=True
        )
        self.width_projection = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=True
        )
        self.apply(_initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        height_context = F.adaptive_avg_pool2d(x, (height, 1))
        width_context = F.adaptive_avg_pool2d(x, (1, width)).transpose(2, 3)
        context = torch.cat((height_context, width_context), dim=2)
        context = self.activation(
            self.normalization(self.shared_projection(context))
        )
        height_context, width_context = torch.split(
            context, (height, width), dim=2
        )
        width_context = width_context.transpose(2, 3)
        height_weight = torch.sigmoid(self.height_projection(height_context))
        width_weight = torch.sigmoid(self.width_projection(width_context))
        return x * height_weight * width_weight


class BipolarCrossInteractionFusion(nn.Module):
    """Fuse decoder and encoder features through agreement and discrepancy."""

    def __init__(
        self,
        channels: int,
        interaction_channels: Optional[int] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        interaction_channels = interaction_channels or max(1, channels // 2)
        fused_channels = 2 * interaction_channels

        self.decoder_projection = nn.Sequential(
            nn.Conv2d(channels, interaction_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(interaction_channels),
            nn.SiLU(inplace=True),
        )
        self.encoder_projection = nn.Sequential(
            nn.Conv2d(channels, interaction_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(interaction_channels),
            nn.SiLU(inplace=True),
        )
        self.interaction_refinement = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                fused_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=fused_channels,
                bias=False,
            ),
            nn.BatchNorm2d(fused_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_projection = nn.Conv2d(
            fused_channels, 1, kernel_size=1, bias=True
        )

        self.apply(_initialize_weights)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

    def forward(self, decoder: Tensor, encoder: Tensor) -> Tensor:
        if decoder.shape != encoder.shape:
            raise ValueError(
                "Decoder and encoder features must have the same shape, got "
                f"{tuple(decoder.shape)} and {tuple(encoder.shape)}."
            )

        decoder_feature = self.decoder_projection(decoder)
        encoder_feature = self.encoder_projection(encoder)
        agreement = decoder_feature * encoder_feature
        discrepancy = torch.abs(decoder_feature - encoder_feature)
        interaction = torch.cat((agreement, discrepancy), dim=1)
        gate = self.gate_projection(self.interaction_refinement(interaction))
        bipolar_gate = 2.0 * torch.sigmoid(gate) - 1.0
        return decoder + encoder * (1.0 + bipolar_gate)


class SRFDepthwiseConvolution(nn.Module):
    """Construct progressive context with serial residual depthwise stages."""

    def __init__(
        self,
        channels: int,
        stages: int = 3,
        activation: str = "relu6",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if stages <= 0:
            raise ValueError("stages must be positive.")

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    _activation(activation),
                )
                for _ in range(stages)
            ]
        )
        self.apply(_initialize_weights)

    def forward(self, x: Tensor) -> List[Tensor]:
        outputs: List[Tensor] = []
        current = x
        for stage in self.stages:
            current = current + stage(current)
            outputs.append(current)
        return outputs


class SRFABlock(nn.Module):
    """Apply channel expansion, SRFDC, adaptive fusion, and residual mapping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: float = 2.0,
        series_stages: int = 3,
        activation: str = "relu6",
    ) -> None:
        super().__init__()
        expanded_channels = int(in_channels * expansion_factor)
        if expanded_channels <= 0:
            raise ValueError("expansion_factor produces no channels.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.series_stages = series_stages

        self.expansion = nn.Sequential(
            nn.Conv2d(
                in_channels, expanded_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            _activation(activation),
        )
        self.serial_depthwise = SRFDepthwiseConvolution(
            expanded_channels,
            stages=series_stages,
            activation=activation,
        )
        self.stage_weight_generator = nn.Conv2d(
            series_stages * expanded_channels,
            series_stages,
            kernel_size=1,
            bias=True,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(
                expanded_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.shuffle_groups = gcd(expanded_channels, out_channels)

        self.apply(_initialize_weights)
        nn.init.zeros_(self.stage_weight_generator.weight)
        nn.init.zeros_(self.stage_weight_generator.bias)

    def _fuse(self, stage_outputs: Sequence[Tensor]) -> Tensor:
        if len(stage_outputs) != self.series_stages:
            raise RuntimeError(
                f"Expected {self.series_stages} SRFDC outputs, "
                f"received {len(stage_outputs)}."
            )
        context = torch.cat(tuple(stage_outputs), dim=1)
        logits = self.stage_weight_generator(context)
        weights = torch.softmax(logits, dim=1) * float(self.series_stages)
        stacked = torch.stack(tuple(stage_outputs), dim=1)
        return torch.sum(stacked * weights.unsqueeze(2), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        expanded = self.expansion(x)
        fused = self._fuse(self.serial_depthwise(expanded))
        fused = _channel_shuffle(fused, self.shuffle_groups)
        return self.skip_projection(x) + self.output_projection(fused)


def _make_stage(
    in_channels: int,
    out_channels: int,
    depth: int,
    expansion_factor: float,
    series_stages: int,
    activation: str,
) -> nn.Sequential:
    if depth <= 0:
        raise ValueError("Each stage depth must be positive.")
    blocks = [
        SRFABlock(
            in_channels,
            out_channels,
            expansion_factor=expansion_factor,
            series_stages=series_stages,
            activation=activation,
        )
    ]
    blocks.extend(
        SRFABlock(
            out_channels,
            out_channels,
            expansion_factor=expansion_factor,
            series_stages=series_stages,
            activation=activation,
        )
        for _ in range(depth - 1)
    )
    return nn.Sequential(*blocks)


class SRFUNet(nn.Module):
    """Five-stage SRF-UNet with selectable small, base, and large widths.

    The network returns a one-element list containing raw segmentation logits,
    matching the interface used for the reported experiments.
    """

    def __init__(
        self,
        size: str = "base",
        num_classes: int = 1,
        in_channels: int = 3,
        channels: Optional[Sequence[int]] = None,
        depths: Sequence[int] = (1, 1, 1, 1, 1),
        series_stages: int = 3,
        expansion_factor: float = 2.0,
        bcif_kernel: int = 3,
        activation: str = "relu6",
    ) -> None:
        super().__init__()
        size_key = _SIZE_ALIASES.get(size.lower())
        if size_key is None:
            choices = ", ".join(MODEL_CHANNELS)
            raise ValueError(f"Unknown model size '{size}'. Choose from: {choices}.")

        selected_channels = tuple(channels or MODEL_CHANNELS[size_key])
        selected_depths = tuple(depths)
        if len(selected_channels) != 5 or any(c <= 0 for c in selected_channels):
            raise ValueError("channels must contain five positive integers.")
        if len(selected_depths) != 5 or any(d <= 0 for d in selected_depths):
            raise ValueError("depths must contain five positive integers.")
        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive.")

        self.size = size_key
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = selected_channels
        self.series_stages = series_stages

        c0, c1, c2, c3, c4 = selected_channels
        encoder_inputs = (in_channels, c0, c1, c2, c3)
        encoder_outputs = (c0, c1, c2, c3, c4)

        self.encoder_attention = nn.ModuleList(
            CoordinateAttention(channels_in) for channels_in in encoder_inputs
        )
        self.encoder = nn.ModuleList(
            _make_stage(
                channels_in,
                channels_out,
                depth,
                expansion_factor,
                series_stages,
                activation,
            )
            for channels_in, channels_out, depth in zip(
                encoder_inputs, encoder_outputs, selected_depths
            )
        )

        decoder_inputs = (c4, c3, c2, c1, c0)
        decoder_outputs = (c3, c2, c1, c0, c0)
        self.decoder_attention = nn.ModuleList(
            CoordinateAttention(channels_in) for channels_in in decoder_inputs
        )
        self.decoder = nn.ModuleList(
            _make_stage(
                channels_in,
                channels_out,
                1,
                expansion_factor,
                series_stages,
                activation,
            )
            for channels_in, channels_out in zip(
                decoder_inputs, decoder_outputs
            )
        )
        self.skip_fusion = nn.ModuleList(
            BipolarCrossInteractionFusion(
                channels_out,
                interaction_channels=max(1, channels_out // 2),
                kernel_size=bcif_kernel,
            )
            for channels_out in (c3, c2, c1, c0)
        )
        self.segmentation_head = nn.Conv2d(c0, num_classes, kernel_size=1)
        _initialize_weights(self.segmentation_head)

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected a four-dimensional NCHW tensor, got shape {tuple(x.shape)}."
            )
        if x.shape[1] == self.in_channels:
            return x
        if x.shape[1] == 1 and self.in_channels == 3:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(
            f"Expected {self.in_channels} input channels, got {x.shape[1]}."
        )

    @staticmethod
    def _upsample(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self._prepare_input(x)
        input_size = x.shape[-2:]

        feature = x
        skips: List[Tensor] = []
        for index, (attention, stage) in enumerate(
            zip(self.encoder_attention, self.encoder)
        ):
            feature = stage(attention(feature))
            feature = F.max_pool2d(feature, kernel_size=2, stride=2)
            if index < 4:
                skips.append(feature)

        for index in range(4):
            feature = self.decoder[index](self.decoder_attention[index](feature))
            skip = skips[-1 - index]
            feature = F.relu(
                self._upsample(feature, skip.shape[-2:]),
                inplace=True,
            )
            feature = self.skip_fusion[index](feature, skip)

        feature = self.decoder[4](self.decoder_attention[4](feature))
        feature = F.relu(self._upsample(feature, input_size), inplace=True)
        return [self.segmentation_head(feature)]


SRF_UNet = SRFUNet


def create_model(size: str = "base", **kwargs) -> SRFUNet:
    """Create an SRF-UNet model from a named width preset."""

    return SRFUNet(size=size, **kwargs)


def srf_unet_s(**kwargs) -> SRFUNet:
    """Create SRF-UNet-S."""

    return create_model("s", **kwargs)


def srf_unet(**kwargs) -> SRFUNet:
    """Create the base SRF-UNet."""

    return create_model("base", **kwargs)


def srf_unet_l(**kwargs) -> SRFUNet:
    """Create SRF-UNet-L."""

    return create_model("l", **kwargs)
