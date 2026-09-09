"""Rank-parameterised building blocks.

PLAN.md rule 2: one builder that picks Conv2d or Conv3d from the rank, not two builders.
Everything a U-shaped network needs is looked up here, so the architecture code never
mentions a dimensionality.

torch has no separable convolution, so it is assembled from a depthwise convolution
(groups = in_channels) followed by a pointwise 1x1.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.common import Activation
from pyplatypus.spec.models import Initialiser


class ModelError(PlatypusError):
    kind = "model_error"


_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
_CONV_TRANSPOSE = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_NORM = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_POOL = {2: nn.MaxPool2d, 3: nn.MaxPool3d}
_DROPOUT = {2: nn.Dropout2d, 3: nn.Dropout3d}
_UPSAMPLE_MODE = {2: "bilinear", 3: "trilinear"}

_ACTIVATION = {
    Activation.RELU: nn.ReLU,
    Activation.LEAKY_RELU: nn.LeakyReLU,
    Activation.ELU: nn.ELU,
    Activation.SELU: nn.SELU,
    Activation.GELU: nn.GELU,
    Activation.SILU: nn.SiLU,
    Activation.TANH: nn.Tanh,
}


def check_rank(rank: int) -> int:
    if rank not in _CONV:
        raise ModelError(f"only 2D and 3D are supported, got rank {rank}")
    return rank


def activation(kind: Activation) -> nn.Module:
    return _ACTIVATION[kind](inplace=True) if kind in {
        Activation.RELU, Activation.LEAKY_RELU, Activation.ELU, Activation.SELU,
        Activation.SILU,
    } else _ACTIVATION[kind]()


def initialise(module: nn.Module, how: Initialiser, kind: Activation) -> None:
    """He for ReLU-like activations, Glorot otherwise - applied to every convolution."""
    gain_nonlinearity = "relu" if kind in {
        Activation.RELU, Activation.LEAKY_RELU, Activation.ELU, Activation.SELU,
        Activation.SILU,
    } else "tanh"
    for layer in module.modules():
        if isinstance(layer, (*_CONV.values(), *_CONV_TRANSPOSE.values())):
            if how is Initialiser.HE_NORMAL:
                nn.init.kaiming_normal_(layer.weight, nonlinearity=gain_nonlinearity)
            elif how is Initialiser.HE_UNIFORM:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=gain_nonlinearity)
            elif how is Initialiser.GLOROT_NORMAL:
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def convolution(rank: int, in_channels: int, out_channels: int, *, kernel_size: int = 3,
                separable: bool = False, bias: bool = True) -> nn.Module:
    conv = _CONV[rank]
    padding = kernel_size // 2
    if not separable:
        return conv(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    return nn.Sequential(
        conv(in_channels, in_channels, kernel_size, padding=padding,
             groups=in_channels, bias=False),
        conv(in_channels, out_channels, 1, bias=bias),
    )


def dropout(rank: int, rate: float, spatial: bool) -> nn.Module:
    """Spatial dropout drops whole feature maps, which is what you want between
    convolutions; ordinary dropout drops individual activations."""
    if rate <= 0:
        return nn.Identity()
    return _DROPOUT[rank](rate) if spatial else nn.Dropout(rate)


def pooling(rank: int) -> nn.Module:
    return _POOL[rank](2)


def upsample(rank: int, in_channels: int, out_channels: int, *, learned: bool = True,
             separable: bool = False) -> nn.Module:
    """Transposed convolution, or interpolation followed by a convolution."""
    if learned:
        return _CONV_TRANSPOSE[rank](in_channels, out_channels, 2, stride=2)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode=_UPSAMPLE_MODE[rank], align_corners=False),
        convolution(rank, in_channels, out_channels, kernel_size=3, separable=separable),
    )


class ConvBlock(nn.Module):
    """`width` convolutions, each optionally normalised, then activated."""

    def __init__(self, rank: int, in_channels: int, out_channels: int, *, width: int = 2,
                 batch_norm: bool = True, separable: bool = False,
                 act: Activation = Activation.RELU, drop: float = 0.0,
                 spatial_dropout: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for index in range(width):
            layers.append(convolution(rank, channels, out_channels,
                                      separable=separable, bias=not batch_norm))
            if batch_norm:
                layers.append(_NORM[rank](out_channels))
            layers.append(activation(act))
            if drop > 0 and index == width - 1:
                layers.append(dropout(rank, drop, spatial_dropout))
            channels = out_channels
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResidualConvBlock(nn.Module):
    """The Res-U-Net block: the same convolutions with a projected identity added back."""

    def __init__(self, rank: int, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.body = ConvBlock(rank, in_channels, out_channels, **kwargs)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else convolution(rank, in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.shortcut(x)
