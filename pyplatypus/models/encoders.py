"""The contracting path, behind a seam.

The decoder consumes anything satisfying `Encoder`: an object that reports the channel
count at each level and returns one feature map per level. Our own encoder is the only
implementation in v0.1; a pretrained ResNet or EfficientNet backbone slots in behind the
same seam, which is the whole reason this file exists separately.

The contract is stated in `Encoder` and **checked by measurement** in `verify_encoder`,
because a declaration can be wrong. Level 0 must be at input resolution: the decoder's
head is a 1x1 convolution on it, so an encoder whose finest feature is already halved -
which is every ImageNet backbone, their stems stride by 2 - produces predictions at half
the size of the mask. Nothing about the shape of that model is suspicious, and the
failure lands much later as a tensor-size error from inside the loss.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

from pyplatypus.models.layers import (
    ConvBlock,
    ModelError,
    ResidualConvBlock,
    check_rank,
    pooling,
)
from pyplatypus.spec.common import Activation


@runtime_checkable
class Encoder(Protocol):
    """Feature maps at descending resolutions.

    `channels` lists the channel count at every level, coarsest last. `forward` returns
    one tensor per entry: levels 0..n-2 are skip connections, the last is the bottleneck.

    Three things the decoder relies on, all enforced by `verify_encoder`:

    1. **Level 0 is at input resolution.** Every later level halves each spatial
       dimension. The head is a 1x1 convolution on level 0, so its resolution is the
       prediction's resolution.
    2. `channels` is accurate. The decoder sizes every one of its own convolutions from
       it before seeing a single tensor.
    3. The number of levels is `blocks + 1`.
    """

    channels: tuple[int, ...]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...


def verify_encoder(encoder: Encoder, rank: int, in_channels: int, levels: int) -> None:
    """Push one tiny tensor through and check what comes back matches what was promised.

    Measuring beats trusting `channels` and a docstring: an encoder that miscounts its
    own widths builds a decoder with the wrong convolutions, and one whose stem strides
    by 2 trains happily against a mask it cannot possibly match. Both surface far from
    their cause. The probe is the smallest input the level count allows, so it costs
    almost nothing even at rank 3.
    """
    declared = tuple(encoder.channels)
    if len(declared) != levels:
        raise ModelError(
            f"encoder reports {len(declared)} levels but the spec asks for {levels} "
            f"(blocks + bottleneck)"
        )

    size = 2 ** (levels - 1)
    probe = torch.zeros(1, in_channels, *([size] * rank))
    call = encoder if isinstance(encoder, nn.Module) else encoder.forward
    was_training = getattr(encoder, "training", False)
    if isinstance(encoder, nn.Module):
        encoder.eval()
    try:
        with torch.no_grad():
            features = call(probe)
    finally:
        if isinstance(encoder, nn.Module) and was_training:
            encoder.train()

    if len(features) != levels:
        raise ModelError(
            f"encoder declares {levels} levels in `channels` but returned "
            f"{len(features)} feature maps; the two must agree"
        )

    got = tuple(int(f.shape[1]) for f in features)
    if got != declared:
        raise ModelError(
            f"encoder declares channels {declared} but returned {got}; the decoder is "
            f"built from the declaration, so the two must agree"
        )

    spatial = [tuple(int(n) for n in f.shape[2:]) for f in features]
    expected_first = tuple([size] * rank)
    if spatial[0] != expected_first:
        raise ModelError(
            f"encoder level 0 is {spatial[0]} for a {expected_first} input, so it "
            f"downsamples before the first skip connection. The decoder's head sits on "
            f"level 0, so predictions would come out at {spatial[0]} against a mask at "
            f"{expected_first}. An ImageNet backbone does this - its stem strides by 2 - "
            f"and needs a full-resolution level prepended."
        )

    for level in range(1, levels):
        wanted = tuple(max(1, n // 2) for n in spatial[level - 1])
        if spatial[level] != wanted:
            raise ModelError(
                f"encoder level {level} is {spatial[level]}, but level {level - 1} was "
                f"{spatial[level - 1]}, so it should be {wanted}; every level must halve "
                f"each spatial dimension"
            )


class UShapedEncoder(nn.Module):
    """Ours: `blocks` convolution blocks, each halving the resolution and doubling width."""

    def __init__(
        self,
        rank: int,
        in_channels: int,
        *,
        blocks: int = 4,
        filters: int = 16,
        residual: bool = False,
        width: int = 2,
        batch_norm: bool = True,
        separable: bool = False,
        act: Activation = Activation.RELU,
        drop: float = 0.0,
        spatial_dropout: bool = True,
    ):
        super().__init__()
        self.rank = check_rank(rank)
        self.blocks = blocks
        block_type = ResidualConvBlock if residual else ConvBlock
        options = {
            "width": width,
            "batch_norm": batch_norm,
            "separable": separable,
            "act": act,
            "drop": drop,
            "spatial_dropout": spatial_dropout,
        }

        self.stages = nn.ModuleList()
        channels = in_channels
        widths = []
        for level in range(blocks):
            out = filters * 2**level
            self.stages.append(block_type(rank, channels, out, **options))
            widths.append(out)
            channels = out

        self.bottleneck = block_type(rank, channels, filters * 2**blocks, **options)
        widths.append(filters * 2**blocks)
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


class PretrainedEncoder(nn.Module):
    """A timm backbone behind the `Encoder` seam, with the levels it cannot supply made
    up by one stage of our own.

    **What is pretrained and what is not.** ImageNet backbones halve the input in their
    stem, so the finest feature they offer is at 1/2 and there is no pretrained content
    at full resolution - there never was, for any of them. Level 0 is therefore a single
    convolution block of ours, trained from scratch, and levels 1..blocks come from the
    backbone. A backbone that does provide a full-resolution level (`vgg16` does) is used
    for level 0 directly.

    Levels are chosen by **downsampling factor, not by position**: the decoder needs
    1, 2, 4, ... 2**blocks, and a backbone whose features start at 1/4 - the patch-based
    ones, `convnext_*` and `swin_*` - cannot supply 1/2 at all. Inventing it would mean
    a made-up level carrying no pretrained information in the place where a U-Net gets
    its fine detail, so those are refused by name instead.

    `filters` sizes the one stage we own; `blocks` says how many of the backbone's stages
    to use. Every other block option applies to our stage only, since the backbone's own
    layers are fixed by whoever trained it.
    """

    #: ImageNet statistics are applied only when the weights that expect them are loaded.
    def __init__(
        self,
        name: str,
        *,
        in_channels: int,
        blocks: int,
        filters: int,
        pretrained: bool = False,
        rank: int = 2,
        **stem_options,
    ):
        super().__init__()
        if rank != 2:
            raise ModelError(
                f"pretrained encoders are 2D only; this model is {rank}D. ImageNet is "
                f"images, so there is nothing to transfer to a volume. Leave `encoder` "
                f"unset to use the built-in encoder, which works at both ranks."
            )
        self.rank = 2
        timm = _import_timm()

        # Created twice on purpose: the first is random-weight and only asked for its
        # feature_info, because which levels a backbone offers has to be known before
        # `out_indices` can be chosen, and choosing wrongly is what this class exists to
        # prevent. The probe downloads nothing.
        try:
            probe = timm.create_model(name, features_only=True, pretrained=False)
        except Exception as error:
            raise ModelError(
                f"'{name}' is not a timm model that can report features: {error}. "
                f"`timm.list_models(pretrained=True)` lists the names; resnet18, "
                f"resnet34, resnet50, efficientnet_b0 and mobilenetv3_large_100 are "
                f"known to work here."
            ) from error

        reductions = list(probe.feature_info.reduction())
        wanted = [2**level for level in range(blocks + 1)]  # 1, 2, 4, ... 2**blocks
        self.own_level_zero = reductions[0] != 1
        needed_from_backbone = wanted[1:] if self.own_level_zero else wanted

        missing = [r for r in needed_from_backbone if r not in reductions]
        if missing:
            raise ModelError(
                f"'{name}' offers features at {reductions}, but this model needs "
                f"{needed_from_backbone}; {missing} is not there. Backbones that pool in "
                f"patches ('convnext_*', 'swin_*') start at 1/4 and cannot supply 1/2, "
                f"which is where a U-shaped decoder recovers fine detail. Use a "
                f"convolutional backbone - resnet*, efficientnet_*, mobilenetv3_* - or "
                f"reduce `blocks` if the deepest level is the one missing."
            )

        indices = tuple(reductions.index(r) for r in needed_from_backbone)
        self.backbone = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
            out_indices=indices,
        )
        backbone_widths = tuple(self.backbone.feature_info.channels())

        if self.own_level_zero:
            self.stem = ConvBlock(2, in_channels, filters, **stem_options)
            self.channels = (filters, *backbone_widths)
        else:
            self.stem = None
            self.channels = backbone_widths

        # A pretrained backbone was fitted on ImageNet-normalised input; windowed medical
        # data is scaled to [0, 1] and nowhere near those statistics. Skipping this
        # throws away much of the transfer and nothing reports it but a worse score.
        # Applied only with the weights that expect it - on a from-scratch run these
        # numbers mean nothing.
        mean, std = self._statistics(self.backbone, in_channels) if pretrained else (None, None)
        self.normalises = mean is not None
        if self.normalises:
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    @staticmethod
    def _statistics(backbone, in_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The backbone's own expected input statistics, widened or narrowed to fit.

        timm adapts the stem's weights when `in_chans` is not 3; the statistics have to
        follow, and a single averaged value is what that adaptation implies.
        """
        config = getattr(backbone, "default_cfg", {}) or {}
        mean = tuple(config.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(config.get("std", (0.229, 0.224, 0.225)))
        if len(mean) != in_channels:
            mean = (sum(mean) / len(mean),) * in_channels
            std = (sum(std) / len(std),) * in_channels
        shape = (1, in_channels, 1, 1)
        return (torch.tensor(mean).reshape(shape), torch.tensor(std).reshape(shape))

    @property
    def transferred(self) -> nn.Module:
        """The part that arrived pretrained, and nothing else.

        `stem` is ours and starts random, so a lower learning rate or a freeze meant to
        protect transferred weights must not touch it - it is the one part that has
        everything to learn.
        """
        return self.backbone

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)] if self.stem is not None else []
        if self.normalises:
            x = (x - self.mean) / self.std
        features.extend(self.backbone(x))
        return features


def _import_timm():
    try:
        import timm
    except ImportError as error:
        raise ModelError(
            "a pretrained encoder needs timm, which is an optional extra:\n"
            "    pip install 'pyplatypus[encoders]'\n"
            "Leave `encoder` unset to use the built-in encoder, which needs nothing "
            "extra and works in 2D and 3D."
        ) from error
    return timm
