"""The U-shaped family: U-Net, U-Net++, Res-U-Net, LinkNet.

The old package expressed these as three independent booleans, so it also built the 4
combinations nobody named and nobody tested. Here `Architecture` picks one, and the
orthogonal switches - separable convolutions, spatial dropout, learned or interpolated
upsampling, deep supervision, block width - compose with all of them.

Nothing below mentions 2D or 3D. Every layer comes from `layers.py`, keyed on rank.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.models.encoders import Encoder, UShapedEncoder, verify_encoder
from pyplatypus.models.layers import (
    ConvBlock,
    ModelError,
    ResidualConvBlock,
    check_rank,
    convolution,
    initialise,
    upsample,
)
from pyplatypus.spec.common import Architecture
from pyplatypus.spec.models import SegmentationModel


class UShapedNet(nn.Module):
    """An encoder, an expanding path, and a 1x1 head per output.

    With `deep_supervision` the forward pass returns a tuple of predictions, one per
    decoder depth and all at input resolution, the final one last. Otherwise a tensor.
    Logits, always - the loss applies its own activation.
    """

    def __init__(self, spec: SegmentationModel, encoder: Encoder | None = None):
        super().__init__()
        self.rank = check_rank(spec.rank)
        self.spec = spec
        self.architecture = spec.architecture
        self.nested = spec.architecture is Architecture.U_NET_PLUS_PLUS
        self.additive = spec.architecture is Architecture.LINKNET
        residual = spec.architecture is Architecture.RES_U_NET

        block_type = ResidualConvBlock if residual else ConvBlock
        self.block_options = {
            "width": spec.block_width, "batch_norm": spec.batch_normalization,
            "separable": spec.separable_conv, "act": spec.activation,
            "drop": spec.dropout, "spatial_dropout": spec.spatial_dropout,
        }

        self.encoder = encoder or UShapedEncoder(
            self.rank, spec.channels, blocks=spec.blocks, filters=spec.filters,
            residual=residual, **self.block_options,
        )
        # Measured, not trusted: a supplied encoder is checked against what the decoder
        # is about to assume of it. Ours passes by construction; the check is here for
        # every other one, and it is cheap.
        verify_encoder(self.encoder, self.rank, spec.channels, spec.blocks + 1)
        widths = self.encoder.channels

        learned = not spec.upsample
        self.ups = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # X[i][j] in the U-Net++ paper. A plain U-Net is the j == blocks - i diagonal of
        # the same grid, so one construction serves both.
        depth = spec.blocks
        for i in range(depth):
            for j in range(1, depth - i + 1):
                if not self.nested and j != depth - i:
                    continue  # plain decoders only walk the diagonal
                self.ups[f"{i}_{j}"] = upsample(
                    self.rank, widths[i + 1], widths[i],
                    learned=learned, separable=spec.separable_conv,
                )
                # LinkNet adds, so width is unchanged. U-Net++ concatenates every
                # earlier node on this row plus the upsampled one; a plain decoder has
                # only the single encoder skip plus the upsampled one.
                if self.additive:
                    incoming = widths[i]
                elif self.nested:
                    incoming = widths[i] * (j + 1)
                else:
                    incoming = widths[i] * 2
                self.decoders[f"{i}_{j}"] = block_type(
                    self.rank, incoming, widths[i], **self.block_options
                )

        self.deep_supervision = spec.deep_supervision
        head_depths = range(1, depth + 1) if spec.deep_supervision else [depth]
        self.heads = nn.ModuleDict({
            str(j): convolution(self.rank, widths[0], spec.n_class, kernel_size=1)
            for j in head_depths
        })

        initialise(self, spec.initialiser, spec.activation)

    def _merge(self, upsampled: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        if self.additive:
            # LinkNet adds instead of concatenating, so widths must already agree.
            out = upsampled
            for skip in skips:
                out = out + skip
            return out
        return torch.cat([*skips, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        features = self.encoder.forward(x) if not isinstance(self.encoder, nn.Module) \
            else self.encoder(x)
        depth = self.spec.blocks

        # grid[i][j]; column 0 is the encoder, the bottleneck is grid[depth][0].
        grid: list[list[torch.Tensor | None]] = [
            [features[i]] + [None] * depth for i in range(depth + 1)
        ]

        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                if not self.nested and j != depth - i:
                    continue
                below = grid[i + 1][j - 1]
                if below is None:
                    continue
                up = self.ups[f"{i}_{j}"](below)
                previous = [grid[i][k] for k in range(j)] if self.nested else [grid[i][0]]
                grid[i][j] = self.decoders[f"{i}_{j}"](self._merge(up, previous))

        outputs = []
        for depth_key in sorted(self.heads, key=int):
            node = grid[0][int(depth_key)]
            if node is None:
                raise ModelError(
                    f"deep supervision asked for output at depth {depth_key}, which this "
                    f"architecture does not produce; it needs u_net_plus_plus"
                )
            outputs.append(self.heads[depth_key](node))

        return tuple(outputs) if self.deep_supervision else outputs[-1]



def build_model(spec: SegmentationModel, encoder: Encoder | None = None) -> UShapedNet:
    """The one entry point. Every architecture, every rank, one call."""
    return UShapedNet(spec, encoder=encoder)
