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


def verify_encoder(
    encoder: Encoder,
    rank: int,
    in_channels: int,
    levels: int | None = None,
    *,
    for_a_decoder: bool = True,
    probe_shape: tuple[int, ...] | None = None,
) -> None:
    """Push one tensor through and check what comes back matches what was promised.

    Measuring beats trusting `channels` and a docstring: an encoder that miscounts its
    own widths builds a decoder with the wrong convolutions, and one whose stem strides
    by 2 trains happily against a mask it cannot possibly match. Both surface far from
    their cause. The probe is the smallest input the level count allows, so it costs
    almost nothing even at rank 3.

    **Two of these checks are the decoder's and not the encoder's**, which is what
    `for_a_decoder` is: level 0 at input resolution and every later level halving are
    requirements of the thing that upsamples back, because the head is a 1x1 convolution
    on level 0. A classifier pools the deepest level and has no use for either, so with
    `False` only the measured agreement between the declaration and reality is checked -
    the part every consumer relies on, since each one sizes something from `channels`
    before seeing a tensor. This is why `convnext_*` and `swin_*` are refused for a
    U-shaped model and admissible for a classifier: the missing 1/2 level is the decoder's
    loss alone.

    `levels` is the count the **caller** requires, and `None` means it has no requirement
    of its own - the classifier reads the deepest width and nothing else, so an encoder
    with three levels or one serves it equally. With a number, a disagreement is refused
    before anything is built from it.

    `probe_shape` is the input to measure with, and a classifier passes its own
    `input_shape` rather than the derived minimum. That is not caution: `swin_*` fixes its
    window to the size it was trained at, so a backbone that cannot run at the size this
    model will actually use says so here, while the specification is being turned into a
    network, rather than on the first batch.
    """
    declared = tuple(encoder.channels)
    if levels is not None and len(declared) != levels:
        raise ModelError(
            f"encoder reports {len(declared)} levels but the spec asks for {levels} "
            f"(blocks + bottleneck)"
        )

    size = 2 ** (len(declared) - 1 if levels is None else levels - 1)
    sides = tuple(probe_shape) if probe_shape is not None else tuple([size] * rank)
    if len(sides) != rank:
        raise ModelError(f"probe_shape has {len(sides)} sides but the model is {rank}D")
    probe = torch.zeros(1, in_channels, *sides)
    call = encoder if isinstance(encoder, nn.Module) else encoder.forward
    was_training = getattr(encoder, "training", False)
    if isinstance(encoder, nn.Module):
        encoder.eval()
    try:
        with torch.no_grad():
            features = call(probe)
    except ModelError:
        raise
    except Exception as error:
        # A backbone that cannot run at this size is the probe's own finding, and timm
        # says so with a bare assertion: `swin_*` fixes the window to the size it was
        # trained at and answers "Input height (64) doesn't match model (224)", which names
        # neither the field to change nor the encoder that minds. Measured 2026-10-27.
        raise ModelError(
            f"the encoder could not run at {sides}: {type(error).__name__}: {error}. Some "
            f"backbones fix the input size they were trained at, so this is refused while "
            f"the network is being built rather than on the first batch - change "
            f"`input_shape` to the size the backbone expects, or name one that does not "
            f"mind."
        ) from error
    finally:
        if isinstance(encoder, nn.Module) and was_training:
            encoder.train()

    if len(features) != len(declared):
        raise ModelError(
            f"encoder declares {len(declared)} levels in `channels` but returned "
            f"{len(features)} feature maps; the two must agree"
        )

    got = tuple(int(f.shape[1]) for f in features)
    if got != declared:
        raise ModelError(
            f"encoder declares channels {declared} but returned {got}; the decoder is "
            f"built from the declaration, so the two must agree"
        )

    if not for_a_decoder:
        return

    spatial = [tuple(int(n) for n in f.shape[2:]) for f in features]
    expected_first = sides
    if spatial[0] != expected_first:
        raise ModelError(
            f"encoder level 0 is {spatial[0]} for a {expected_first} input, so it "
            f"downsamples before the first skip connection. The decoder's head sits on "
            f"level 0, so predictions would come out at {spatial[0]} against a mask at "
            f"{expected_first}. An ImageNet backbone does this - its stem strides by 2 - "
            f"and needs a full-resolution level prepended."
        )

    for level in range(1, len(declared)):
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

    **`bottleneck_only` is the classifier's shape of this**, and it is the same seam with
    one level instead of a pyramid: the deepest features the backbone offers, no stage of
    ours in front, nothing made up. Then `blocks` and `filters` describe things that do not
    exist here and are refused rather than ignored - there is no pyramid to size and no
    stage of ours to widen - and the refusal above does not apply at all, because the 1/2
    level a patch-based backbone cannot supply is the decoder's loss and a classifier has
    no decoder. `convnext_*` and `swin_*` are admissible here for exactly that reason.
    """

    #: ImageNet statistics are applied only when the weights that expect them are loaded.
    def __init__(
        self,
        name: str,
        *,
        in_channels: int,
        blocks: int | None = None,
        filters: int | None = None,
        pretrained: bool = False,
        rank: int = 2,
        bottleneck_only: bool = False,
        **stem_options,
    ):
        super().__init__()
        self.bottleneck_only = bottleneck_only
        if bottleneck_only:
            named = [
                field
                for field, value in (("blocks", blocks), ("filters", filters))
                if value is not None
            ]
            if named:
                joined = " and ".join("`" + field + "`" for field in named)
                raise ModelError(
                    f"{joined} describe a pyramid and the full-resolution stage in front of "
                    f"it, and `bottleneck_only` builds neither - it takes the deepest "
                    f"features '{name}' offers and nothing else. Drop them, or drop "
                    f"`bottleneck_only` to build the pyramid a decoder needs."
                )
        elif blocks is None or filters is None:
            raise ModelError(
                "a pyramid needs `blocks`, to say how many of the backbone's stages to "
                "use, and `filters`, to size the full-resolution stage in front of them. "
                "Pass `bottleneck_only=True` for the deepest level alone, which is what a "
                "classifier pools."
            )
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
        if not reductions:
            raise ModelError(f"'{name}' reports no features at all, so there is nothing to use")

        if bottleneck_only:
            # The deepest level, by reduction rather than by position: the two agree for
            # every backbone seen here, and the question being asked is which features are
            # the coarsest, which is what `max` says and what `[-1]` assumes.
            self.own_level_zero = False
            indices = (reductions.index(max(reductions)),)
        else:
            wanted = [2**level for level in range(blocks + 1)]  # 1, 2, 4, ... 2**blocks
            self.own_level_zero = reductions[0] != 1
            needed_from_backbone = wanted[1:] if self.own_level_zero else wanted

            missing = [r for r in needed_from_backbone if r not in reductions]
            if missing:
                raise ModelError(
                    f"'{name}' offers features at {reductions}, but a U-shaped decoder needs "
                    f"{needed_from_backbone}; {missing} is not there. Backbones that pool in "
                    f"patches ('convnext_*', 'swin_*') start at 1/4 and cannot supply 1/2, "
                    f"which is where a U-shaped decoder recovers fine detail. Use a "
                    f"convolutional backbone - resnet*, efficientnet_*, mobilenetv3_* - or "
                    f"reduce `blocks` if the deepest level is the one missing. A classifier "
                    f"pools the deepest level and needs none of this, so the same backbone "
                    f"is admissible with `bottleneck_only`."
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

        # **Asked rather than guessed, and measured when it was written.** timm's
        # transformer backbones return features channels-last: `swin_tiny` answers
        # (1, 7, 7, 768) where `convnext_tiny`, `resnet34` and `vit_small` answer
        # (1, 768, 7, 7). The seam here is channels-first, so the layout is read off
        # the wrapper's own `output_fmt` - `Format.NHWC` against `Format.NCHW` - and
        # not from the shapes, which cannot be told apart when the channel count
        # happens to equal the grid. Passing `output_fmt='NCHW'` to `create_model` is
        # not the fix: swin accepts it and returns channels-last anyway, and every
        # convolutional backbone refuses the argument outright. A wrong answer here is
        # caught loudly by `verify_encoder`, which compares the declared channels
        # against the axis this class puts them on.
        self.channels_last = "NHWC" in str(getattr(self.backbone, "output_fmt", "NCHW"))

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

    def _channels_first(self, feature: torch.Tensor) -> torch.Tensor:
        """A channels-last feature map, transposed once at the seam.

        Here rather than in every consumer: the decoder, the pooled head and
        `verify_encoder` would otherwise each need to know which backbones timm reports
        channels-last, which is three places for one fact.
        """
        if not self.channels_last:
            return feature
        axes = (0, feature.dim() - 1, *range(1, feature.dim() - 1))
        return feature.permute(*axes).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)] if self.stem is not None else []
        if self.normalises:
            x = (x - self.mean) / self.std
        features.extend(self._channels_first(f) for f in self.backbone(x))
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
