"""Turning mask files into class indices, and predictions back into pictures.

The colormap is the contract: position in the list is the class index, and the colour is
what that class looks like on disk. Everything here is rank-agnostic - nothing indexes
`[:, :]`, so a volume works the same as an image.
"""

from __future__ import annotations

import numpy as np

from pyplatypus.errors import PlatypusError


class MaskError(PlatypusError):
    kind = "mask_error"


def unite_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Combine several mask files covering one image into one array.

    The Data Science Bowl stores one file per nucleus; taking the element-wise maximum
    merges them. A single multi-class mask file passes through untouched.
    """
    if not masks:
        raise MaskError("no masks to unite")
    first = masks[0]
    for index, mask in enumerate(masks[1:], start=1):
        if mask.shape != first.shape:
            raise MaskError(
                f"mask {index} has shape {mask.shape} but mask 0 has {first.shape}; "
                "masks belonging to one image must be the same size"
            )
    return np.maximum.reduce(masks)


def colours_to_classes(mask: np.ndarray, colormap: list[tuple[int, int, int]],
                       *, tolerance: int = 0) -> tuple[np.ndarray, float]:
    """Map an RGB mask to class indices, dropping the channel axis.

    Anything matching no colour becomes class 0. `unmatched_fraction` exists so a caller
    can notice when that is happening to most of the image, which usually means the
    colormap does not describe this dataset.
    """
    if mask.ndim < 2:
        raise MaskError(f"a mask needs at least 2 dimensions, got shape {mask.shape}")
    if mask.shape[-1] < 3:
        raise MaskError(
            f"expected an RGB mask with 3 channels last, got shape {mask.shape}"
        )
    rgb = mask[..., :3].astype(np.int16)

    classes = np.zeros(rgb.shape[:-1], dtype=np.int64)
    matched = np.zeros(rgb.shape[:-1], dtype=bool)
    # Later colours win, so an explicit class beats the background it overlaps.
    for index, colour in enumerate(colormap):
        target = np.asarray(colour, dtype=np.int16)
        if tolerance:
            hit = np.all(np.abs(rgb - target) <= tolerance, axis=-1)
        else:
            hit = np.all(rgb == target, axis=-1)
        classes[hit] = index
        matched |= hit
    return classes, float(1.0 - matched.mean())


def labels_to_classes(mask: np.ndarray, labels: list[int], *, tolerance: float = 0.5
                      ) -> tuple[np.ndarray, float]:
    """Map a label map to class indices, dropping the channel axis.

    The other half of `colours_to_classes`, for masks that hold numbers rather than
    pictures - which is how every volume format labels anything, and how a single-channel
    PNG can be read as well.

    Compared with a tolerance because the values arrive as floats: a label map that has been
    resampled, or merely round-tripped through float32, will not satisfy `== 2` reliably,
    and a label silently failing to match becomes background. Half a unit is the right
    tolerance for integer labels and catches nothing else.
    """
    if mask.ndim < 2:
        raise MaskError(f"a mask needs at least 2 dimensions, got shape {mask.shape}")
    values = mask[..., 0] if mask.shape[-1] == 1 else mask
    if values.shape != mask.shape[:-1]:
        raise MaskError(
            f"a label map must have one channel, got shape {mask.shape}. A mask with "
            "several channels is a picture - use a colormap for it."
        )

    classes = np.zeros(values.shape, dtype=np.int64)
    matched = np.zeros(values.shape, dtype=bool)
    for index, label in enumerate(labels):
        hit = np.abs(values - float(label)) <= tolerance
        classes[hit] = index
        matched |= hit
    return classes, float(1.0 - matched.mean())


def signed_distance(onehot: np.ndarray, spacing: tuple[float, ...] | None = None
                    ) -> np.ndarray:
    """Distance to the nearest boundary, negative inside each class and positive outside.

    One map per class, same shape as the mask it came from, for a loss that needs to know
    *how far* a prediction is wrong rather than only that it is. Dice counts a voxel the
    same wherever it sits, which is why a model can reach 0.88 on it while reporting
    volumes a fifth too large - the overshoot is all at the boundary, and the boundary is
    most of a small lesion.

    **`spacing` makes it millimetres rather than voxels**, and giving it is the difference
    between a loss that means the same thing on two scanners and one that does not: a slice
    2.5 mm thick and one 1 mm thick put the same anatomy at different voxel distances.
    Unset, the axes are treated as equal, which is what a 2D image with no physical scale
    wants.

    A class that is absent from a mask has no boundary, so its map is left at zero: there
    is nothing to be near or far from, and any other filling would be a number the loss
    would then act on.
    """
    from scipy.ndimage import distance_transform_edt

    array = np.asarray(onehot)
    if array.ndim < 2:
        raise MaskError(
            f"expected a one-hot mask with a class axis last; got shape {array.shape}"
        )
    out = np.zeros(array.shape, dtype=np.float32)
    for index in range(array.shape[-1]):
        inside = array[..., index] > 0.5
        if not inside.any() or inside.all():
            continue
        out[..., index] = (
            distance_transform_edt(~inside, sampling=spacing)
            - distance_transform_edt(inside, sampling=spacing)
        )
    return out


def classes_to_onehot(classes: np.ndarray, n_class: int) -> np.ndarray:
    """Class indices to a channels-last one-hot array."""
    highest = int(classes.max(initial=0))
    if highest >= n_class:
        raise MaskError(
            f"found class index {highest} but only {n_class} classes are defined"
        )
    return np.eye(n_class, dtype=np.float32)[classes]


def onehot_to_colours(onehot: np.ndarray, colormap: list[tuple[int, int, int]]
                      ) -> np.ndarray:
    """A predicted one-hot (or probability) array back to an RGB picture."""
    if onehot.shape[-1] != len(colormap):
        raise MaskError(
            f"prediction has {onehot.shape[-1]} channels but the colormap defines "
            f"{len(colormap)} classes"
        )
    return np.asarray(colormap, dtype=np.uint8)[onehot.argmax(axis=-1)]
