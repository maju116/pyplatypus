"""One answer per image: an encoder, pooled, and a dense layer.

This is the smallest network in the package and the one with the fewest decisions in it,
which is the point - everything below the pooling already existed. `UShapedEncoder` is the
contracting path a U-Net has always built, `PretrainedEncoder` is the same timm seam with
`bottleneck_only` instead of a pyramid, and what this file adds is the three lines after
them: pool, drop, project.

**The refusals that matter here are not here.** A classification specification is refused
for an overlap loss, for our knobs beside somebody else's backbone and for a side that runs
out of pixels while it is being *read*, in `spec/classification.py`, so nothing in this file
has to be defensive about a configuration. What is left to check is the encoder, and that is
measured rather than trusted: `verify_encoder` is asked for the agreement between what the
encoder declares and what it returns, because the head is sized from the declaration.

**Two of `verify_encoder`'s checks are deliberately not asked for.** Level 0 at input
resolution and every level halving are the decoder's requirements - its head is a 1x1
convolution on level 0 - and a classifier pools the deepest level. That is the whole reason
`convnext_*` and `swin_*` can be used here and are refused for a U-shaped model: the 1/2
level they cannot supply is where a decoder recovers fine detail, and this network has no
decoder to recover anything in.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.models.encoders import Encoder, PretrainedEncoder, UShapedEncoder, verify_encoder
from pyplatypus.models.layers import check_rank, global_pool, initialise
from pyplatypus.spec.classification import ClassificationModel


def _encoder_for(spec: ClassificationModel) -> Encoder:
    """The contracting path the specification asks for.

    Resolved here rather than by the caller, because for this task `encoder` **is** the
    architecture - there is no second field naming the network - so a builder that ignored
    it would quietly build something the specification does not describe.

    Ours is the plain convolution stack rather than the residual one. A classifier has no
    `architecture` field to choose with, and inventing a `residual` flag would be a new knob
    for a network that already has the honest answer to it: `encoder: resnet34` is the
    residual network, pretrained, and it is one word.
    """
    if spec.encoder is None:
        return UShapedEncoder(
            spec.rank,
            spec.channels,
            blocks=spec.blocks,
            filters=spec.filters,
            residual=False,
            width=spec.block_width,
            batch_norm=spec.batch_normalization,
            separable=spec.separable_conv,
            act=spec.activation,
            drop=spec.dropout,
            spatial_dropout=spec.spatial_dropout,
        )
    return PretrainedEncoder(
        spec.encoder,
        in_channels=spec.channels,
        pretrained=spec.pretrained,
        rank=spec.rank,
        bottleneck_only=True,
    )


class PooledClassifier(nn.Module):
    """Logits for one label per image, always - the loss applies its own activation.

    `n_class` is passed rather than read off the model for the reason `build_model` gives:
    the data decides how many classes there are, and a model that also declared it would be
    a second place for the same fact.
    """

    def __init__(self, spec: ClassificationModel, *, n_class: int, encoder: Encoder | None = None):
        super().__init__()
        self.rank = check_rank(spec.rank)
        self.spec = spec
        self.architecture = spec.architecture
        self.encoder = encoder or _encoder_for(spec)

        # `levels=None`: the number of levels is not this network's requirement. It reads
        # the deepest width and discards the rest, so an encoder with one level and one
        # with five serve it equally - asserting a count would be asserting more than the
        # consumer relies on, which is the opposite of the mistake `#294` was.
        #
        # The probe runs at `input_shape` only for a backbone, and that is not caution:
        # `swin_*` fixes its window to the size it was trained at, so a backbone that
        # cannot run at this model's size says so while the network is being built rather
        # than on the first batch. Ours works at any size the spec already permits, so it
        # keeps the cheap derived probe.
        verify_encoder(
            self.encoder,
            self.rank,
            spec.channels,
            for_a_decoder=False,
            probe_shape=tuple(spec.input_shape) if spec.encoder is not None else None,
        )

        self.pool = global_pool(self.rank, spec.pooling)
        self.drop = nn.Dropout(spec.head_dropout) if spec.head_dropout else nn.Identity()
        self.head = nn.Linear(self.encoder.channels[-1], n_class)

        # Everything but the transfer, which is `#294`'s lesson in the file that would
        # otherwise have repeated it: a backbone arrives with weights and `initialise`
        # walks every layer under `self`.
        initialise(
            self,
            spec.initialiser,
            spec.activation,
            except_for=getattr(self.encoder, "transferred", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = (
            self.encoder(x) if isinstance(self.encoder, nn.Module) else self.encoder.forward(x)
        )
        pooled = self.pool(features[-1])
        return self.head(self.drop(torch.flatten(pooled, 1)))


def build_classifier(
    spec: ClassificationModel, *, n_class: int, encoder: Encoder | None = None
) -> PooledClassifier:
    """The one entry point, beside `build_model` and shaped like it."""
    return PooledClassifier(spec, n_class=n_class, encoder=encoder)
