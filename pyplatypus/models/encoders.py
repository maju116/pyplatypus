"""The contracting path, behind a seam.

The decoder consumes anything satisfying `Encoder`: an object that reports the channel
count at each level and returns one feature map per level. Our own encoder is the only
implementation in v0.1; a pretrained ResNet or EfficientNet backbone slots in later
without the decoder changing, which is the whole reason this file exists separately.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

from pyplatypus.models.layers import ConvBlock, ResidualConvBlock, check_rank, pooling
from pyplatypus.spec.common import Activation


@runtime_checkable
class Encoder(Protocol):
    """Feature maps at descending resolutions.

    `channels` lists the channel count at every level, coarsest last. `forward` returns
    one tensor per entry: levels 0..n-2 are skip connections, the last is the bottleneck.
    """

    channels: tuple[int, ...]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...


class UShapedEncoder(nn.Module):
    """Ours: `blocks` convolution blocks, each halving the resolution and doubling width."""

    def __init__(self, rank: int, in_channels: int, *, blocks: int = 4, filters: int = 16,
                 residual: bool = False, width: int = 2, batch_norm: bool = True,
                 separable: bool = False, act: Activation = Activation.RELU,
                 drop: float = 0.0, spatial_dropout: bool = True):
        super().__init__()
        self.rank = check_rank(rank)
        self.blocks = blocks
        block_type = ResidualConvBlock if residual else ConvBlock
        options = {"width": width, "batch_norm": batch_norm, "separable": separable,
                   "act": act, "drop": drop, "spatial_dropout": spatial_dropout}

        self.stages = nn.ModuleList()
        channels = in_channels
        widths = []
        for level in range(blocks):
            out = filters * 2 ** level
            self.stages.append(block_type(rank, channels, out, **options))
            widths.append(out)
            channels = out

        self.bottleneck = block_type(rank, channels, filters * 2 ** blocks, **options)
        widths.append(filters * 2 ** blocks)
        self.channels = tuple(widths)
        self.pool = pooling(rank)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            x = self.pool(x)
        features.append(self.bottleneck(x))
        return features
