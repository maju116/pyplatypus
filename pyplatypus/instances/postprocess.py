"""Turning masks into instances: the naive way, and the way with a separation signal.

`instances_from_mask` is **a yardstick, not a contribution.** Connected components is what the
published `dsbowl-unet` weights can be scored with at all, and their card already says why it
will do badly: *semantic, not instance* - touching nuclei come back as one region, and one
region is one instance here. The number it produces is the baseline every model in this
milestone is measured against, and it quantifies a limitation the documentation has so far
only named.

`parts_from_instances` and `instances_from_parts` are the two halves of the alternative
(#159): decompose each object into an interior and the rim the erosion removed, predict those
as three mutually exclusive classes, and grow the interiors back. The pair is written before
any model because **the post-process is what caps the approach**, and feeding it the ground
truth measures that cap without training anything.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pyplatypus.instances.metrics import InstanceError

#: 4-connectivity, not 8. Two nuclei touching only at a corner are two objects under 4 and
#: one under 8, and a diagonal join is the cheapest way to merge things that are not joined.
#: It makes this baseline slightly kinder, which is the direction to err for a yardstick the
#: later work has to beat.
CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def instances_from_mask(mask: np.ndarray, minimum_area: int = 0) -> np.ndarray:
    """Label every connected foreground region of a binary or class-index mask.

    `minimum_area` drops regions below a pixel count and relabels so the result stays
    consecutive. Zero by default: a floor is a tuning knob, and a baseline that has been
    tuned is no longer the thing later work gets compared against.
    """
    if mask.ndim != 2:
        raise InstanceError(
            f"expected a 2-D mask, got shape {mask.shape}. Instances in volumes are a "
            "separate question from instances in images and this does not answer it"
        )
    labels, found = ndimage.label(mask > 0, structure=CONNECTIVITY)
    return _drop_small(labels, found, minimum_area)


def _drop_small(labels: np.ndarray, found: int, minimum_area: int) -> np.ndarray:
    """Remove regions below a pixel count and relabel so the result stays consecutive."""
    if minimum_area <= 0 or found == 0:
        return labels.astype(np.int32)
    areas = np.bincount(labels.ravel())
    keep = np.nonzero(areas[1:] >= minimum_area)[0] + 1
    lookup = np.zeros(found + 1, dtype=np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return lookup[labels]


def _neighbours_agree(labels: np.ndarray, *, across_background: bool) -> np.ndarray:
    """Where every 4-neighbour of a labelled pixel carries the same label.

    One vectorised pass instead of eroding each object in turn, which matters: one Data
    Science Bowl image holds 375 objects, and 375 full-image erosions to shave one pixel off
    each is work proportional to the wrong thing.

    `across_background` decides whether touching background counts as disagreement. True
    erodes an object everywhere, which is the ordinary interior; False erodes it only where a
    *different object* touches, which is the weaker form kept for objects too thin to survive
    the first - see `parts_from_instances`.

    **Padded by edge replication, not with zeros.** A zero pad would make the image's own
    border look like background, so an object running off the edge of the picture would be
    given a rim there - a rim that separates it from nothing, costing interior for no gain.
    """
    padded = np.pad(labels, 1, mode="edge")
    agree = labels > 0
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        neighbour = padded[dy : dy + labels.shape[0], dx : dx + labels.shape[1]]
        same = neighbour == labels
        if not across_background:
            same |= neighbour == 0
        agree &= same
    return agree


def parts_from_instances(labels: np.ndarray, width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """An instance label map as an interior and the rim the erosion removed.

    Returns `(interior, rim)`, both boolean and disjoint, whose union is the foreground. With
    the background they are three mutually exclusive classes, which is the whole point: the
    model that predicts them needs nothing this package does not already have, because that
    is ordinary semantic segmentation on the softmax path.

    Touching objects stop touching because each one loses its own rim, which is what makes
    the interiors separable by connected components when the objects were not.

    **An object too thin to survive the erosion is handled rather than ignored**, and this is
    the part that would otherwise be a silent defect. A nucleus one or two pixels wide erodes
    to nothing, and an object absent from the target is an object the model is being told not
    to find - it would be a false negative in every epoch and in the score, with no sign that
    the target was at fault rather than the prediction. So an object whose interior comes back
    empty falls back to the weaker erosion that only removes pixels touching a *different*
    object; one still empty after that keeps itself whole. The fallback can leave two such
    objects sharing a seed, which is a merge - but a merge of two objects that could not be
    represented at all is the better of the two errors, and it is rare: on the Data Science
    Bowl it is measured rather than guessed at.
    """
    if labels.ndim != 2:
        raise InstanceError(
            f"expected a 2-D label map, got shape {labels.shape}. Instances in volumes are a "
            "separate question from instances in images and this does not answer it"
        )
    if width < 1:
        raise InstanceError(
            f"width is {width}; an erosion of less than one pixel removes no rim, so the "
            "interiors would still touch and this whole decomposition would do nothing"
        )

    labels = np.asarray(labels)
    interior = labels > 0
    for _ in range(width):
        interior &= _neighbours_agree(np.where(interior, labels, 0), across_background=True)

    present = np.unique(labels[labels > 0])
    survived = np.unique(labels[interior])
    lost = np.setdiff1d(present, survived, assume_unique=True)
    if lost.size:
        thin = np.isin(labels, lost)
        # The whole label map, not just the thin objects. Masking the others out first was
        # the first version of this and it was wrong in the quiet direction: a neighbour that
        # had been zeroed reads as background, `across_background=False` counts background as
        # agreement, and the pixel touching another object was kept - handing the thin object
        # a seed it shared with its neighbour, which is the one thing this fallback exists to
        # avoid. Caught by a test, not by reading.
        weak = thin & _neighbours_agree(labels, across_background=False)
        # Still nothing means an object wedged between others on every side; keep it whole
        # rather than drop it, because a target cannot ask for an object it does not contain.
        kept = np.unique(labels[weak])
        hopeless = np.isin(labels, np.setdiff1d(lost, kept, assume_unique=True))
        interior |= weak | hopeless

    return interior, (labels > 0) & ~interior


def instances_from_parts(
    interior: np.ndarray, foreground: np.ndarray, minimum_area: int = 0
) -> np.ndarray:
    """Instances from a predicted interior and a predicted foreground.

    The interiors are labelled as seeds and every other foreground pixel joins **the nearest
    seed**, which `scipy.ndimage.distance_transform_edt` answers in one call: the distance
    transform of `seeds == 0` carries, for every pixel, the coordinates of the closest pixel
    where a seed is. That is a Voronoi partition of the foreground, and it needs no dependency
    this package does not have.

    Not a watershed, and the difference is worth stating. A watershed would follow the image's
    own gradient, so it can place a join off the geometric midline where the picture says the
    objects part. This splits on distance alone. For nuclei - small, roughly convex, meeting
    along a short seam - the two agree closely; where objects are long or interlocking they do
    not, and that is a reason to revisit this rather than an argument that it is wrong here.
    `scikit-image` is not a dependency and is not worth adding for one function;
    `scipy.ndimage.watershed_ift` is available but wants integer markers and implements a
    different algorithm from the one its name suggests.

    A foreground pixel that no seed can reach - because the prediction found no interior
    anywhere - stays background rather than becoming an instance of its own. That is the
    honest answer: the model said "object here" and "no object centre anywhere", and inventing
    a centre would turn a contradiction into a confident wrong count.
    """
    interior = np.asarray(interior)
    foreground = np.asarray(foreground)
    if interior.ndim != 2:
        raise InstanceError(f"expected a 2-D interior, got shape {interior.shape}")
    if interior.shape != foreground.shape:
        raise InstanceError(
            f"interior is {interior.shape} and foreground is {foreground.shape}; they are two "
            "readings of one picture and must be the same size"
        )

    seeds, found = ndimage.label(interior > 0, structure=CONNECTIVITY)
    if found == 0:
        return np.zeros(interior.shape, dtype=np.int32)

    _, indices = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    grown = seeds[tuple(indices)]
    grown[foreground <= 0] = 0
    return _drop_small(grown, found, minimum_area)
