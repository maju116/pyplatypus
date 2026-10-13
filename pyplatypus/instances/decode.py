"""One model's output, read back into objects the way its `decode:` block asks.

The other half of `targets.py`. That module says what the channels mean going in; this one
reads the same contract coming out, which is why the two are a pair and why neither
restates the channel order in prose - `Separation.semantic_from` is the one number both of
them ask.

**Decoding is cheap and that is the reason `decode` is a list.** Measured on the Data
Science Bowl's 134 held-out cases: nearest-seed costs 1.41 s for all of them and connected
components 0.10 s, against 15.5 s for one training epoch and 16 minutes for a second
training run. So several decodings of one forward pass is `metrics:`-shaped rather than
`callbacks:`-shaped, and `evaluate` returns a row per (model, decoding) instead of asking
anybody to train twice.

**What is not here is a choice of surface.** `watershed` floods the model's own rim
probability, because that is the one defensible answer: not the image - a nucleus is bright
and so is the next one - and not the distance transform, which is what a ground-truth
ceiling run has to use because perfect parts give a flat surface and nothing to climb.
`pyplatypus#211` measured that choice as worth +0.0132 to the watershed and nothing to the
outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyplatypus.instances.flows import instances_by_flows
from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.postprocess import (
    instances_by_watershed,
    instances_from_mask,
    instances_from_parts,
)
from pyplatypus.spec.instances import Separation

#: The class indices `instance_target` writes for `rim`, which is the contract this reads
#: back. Written down once here rather than as a 1 and a 2 inside three functions.
INTERIOR, RIM = 1, 2


@dataclass(frozen=True)
class Reading:
    """What a decoding needs out of one prediction, with nothing a decoding does not.

    `interior` and `surface` are None for a `flows` prediction and `flow` is None for a
    `rim` one, which is the asymmetry the pair-refusal in the specification exists to keep
    out of here: by the time a decoding runs, the combination has already been refused by
    name, so a None reaching a decoding is a defect rather than a user's mistake.
    """

    foreground: np.ndarray
    interior: np.ndarray | None = None
    surface: np.ndarray | None = None
    flow: np.ndarray | None = None


def read_prediction(prediction: np.ndarray, separation: Separation) -> Reading:
    """Split one channels-last prediction into the parts a decoding reads.

    Args:
        prediction: `(H, W, C)` as the engine's `predict` returns it - the class channels
            activated, the vector channels **raw**. A softmax over a flow component is a
            number and a gradient and no error anywhere, so the two blocks are activated
            differently and that difference arrives here already made.
        separation: which target the model was trained on.

    Returns:
        A `Reading`.

    >>> rim = np.zeros((4, 4, 3), dtype=np.float32)
    >>> rim[..., 0] = 1.0            # all background
    >>> rim[1:3, 1:3, 0], rim[1:3, 1:3, 1] = 0.0, 1.0   # a 2x2 interior
    >>> reading = read_prediction(rim, Separation.RIM)
    >>> int(reading.foreground.sum()), int(reading.interior.sum())
    (4, 4)
    """
    prediction = np.asarray(prediction)
    if prediction.ndim != 3:
        raise InstanceError(
            f"expected one channels-last prediction (H, W, C), got shape {prediction.shape}"
        )
    classes = np.argmax(prediction[..., separation.semantic_from :], axis=-1)
    foreground = classes > 0

    if separation is Separation.RIM:
        return Reading(
            foreground=foreground,
            interior=classes == INTERIOR,
            # The rim's own probability, which is high exactly where the network thinks two
            # objects part. `nearest_seed` throws this away; the watershed is the decoding
            # that reads it.
            #
            # **No test catches this channel being wrong and that is said rather than
            # papered over.** Swapping it for the interior's probability survives the whole
            # suite, and the reason is structural rather than a gap in the fixtures: every
            # unit test here decodes a *perfect* target, where the surface is binary and
            # the rim band is symmetric, so a flood from the interiors lands on the same
            # seam whichever of the two it climbs - measured, IoU 1.0 for both. It takes a
            # soft surface, which takes a trained model. `#211` is where that measurement
            # lives: a perfect surface is worth +0.0132 to this decoding against its own
            # prediction, ten times the 0.0013 two runs differ by, and the same issue is
            # where the ceiling caveat comes from - perfect parts give the watershed
            # nothing to climb.
            surface=prediction[..., RIM].astype(np.float32),
        )
    return Reading(
        foreground=foreground,
        flow=np.moveaxis(prediction[..., :2], -1, 0).astype(np.float32),
    )


def decode_instances(prediction: np.ndarray, separation: Separation, block) -> np.ndarray:
    """A label map from one prediction, by one `decode:` block.

    Args:
        prediction: one channels-last prediction, as `read_prediction` describes.
        separation: which target the model was trained on.
        block: one entry of a model's `decode` list.

    Returns:
        `(H, W)` int32, objects numbered from 1.

    Raises:
        InstanceError: if the block reads a signal this separation does not produce. The
            specification refuses that pair when it is read, so reaching this means the
            decoding was assembled by hand rather than taken from a model.
    """
    reading = read_prediction(prediction, separation)
    if block.needs and separation not in block.needs:
        raise InstanceError(
            f"decoding '{block.name}' reads a signal that separation "
            f"'{separation.value}' does not produce"
        )

    if block.name == "connected_components":
        return instances_from_mask(reading.foreground, minimum_area=block.minimum_area)
    if block.name == "nearest_seed":
        return instances_from_parts(
            reading.interior, reading.foreground, minimum_area=block.minimum_area
        )
    if block.name == "watershed":
        return instances_by_watershed(
            reading.interior,
            reading.foreground,
            reading.surface,
            minimum_area=block.minimum_area,
        )
    if block.name == "follow_flows":
        return instances_by_flows(
            reading.flow,
            reading.foreground,
            steps=block.steps,
            step_size=block.step_size,
            minimum_area=block.minimum_area,
        )
    raise InstanceError(
        f"no decoding called '{block.name}'. The specification's union is the list of them, "
        "so reaching here means a block was added there and not here"
    )
