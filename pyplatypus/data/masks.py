"""Turning mask files into class indices, and predictions back into pictures.

The colormap is the contract: position in the list is the class index, and the colour is
what that class looks like on disk. Everything here is rank-agnostic - nothing indexes
`[:, :]`, so a volume works the same as an image.

`save_class_masks` is the one thing here that touches disk, and it is here rather than
beside the readers because it is what `onehot_to_colours`'s refusal points at: the pair is
the point. Overlapping classes cannot be one picture, so they are one picture each.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

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


def masks_to_codes(masks: list[np.ndarray], *, threshold: float = 0.0) -> tuple[np.ndarray, float]:
    """One binary file per class, as one integer per voxel whose bits say which classes.

    `classes_from`'s reader, and the one naming whose files **cannot be united**:
    `unite_masks` takes an element-wise maximum, which is precisely where per-class
    identity is destroyed - a vessel crossing an organ, or two annotators' masks, become
    one blob and nothing downstream can tell there were two. A bit per class keeps them
    apart while staying *one integer per voxel*, which is what lets tiling and augmentation
    stay untouched: both operate on a single-channel array, and `codes_to_onehot` expands in
    the last step before the loss, the way `region_vocabulary`'s table does.

    **A voxel is in a class when its file is above `threshold`, and the default is zero
    rather than a midpoint.** A midpoint is the obvious rule and it is wrong here: these
    files are binary and arrive as both 0/255 pictures and 0/1 label maps, and `>= 128`
    reads every voxel of the second kind as background - a whole dataset silently empty.
    Masks are read nearest, so nothing between the two values was invented by this package;
    what a lossy encoder left around an edge is included, which is under 1% of a correct
    mask and visible in `unmatched` rather than hidden.

    Lossless to 62 classes in int64, and refused above that rather than wrapping into a
    class that was never asked for.

    Returns the codes and the fraction of voxels in no class at all - the same quantity the
    other three readers return, meaning the same thing. With independent classes there is no
    background channel, so on a small target it sits near 1, which is why the engine's
    warning about that number is suppressed for every multilabel naming rather than for
    `regions` alone.
    """
    if not masks:
        raise MaskError("no masks to code")
    if len(masks) > 62:
        raise MaskError(
            f"{len(masks)} classes is more than one int64 can carry a bit for; the coded "
            "form holds 62. Past that the target has to be built channels-first, which is "
            "a different pipeline rather than a bigger number."
        )
    first = np.asarray(masks[0])
    codes = np.zeros(first.shape[:-1] if first.shape[-1] == 1 else first.shape, dtype=np.int64)
    hit = np.zeros(codes.shape, dtype=bool)
    for position, mask in enumerate(masks):
        array = np.asarray(mask)
        plane = array[..., 0] if array.shape[-1] == 1 else array
        if plane.shape != codes.shape:
            raise MaskError(
                f"class {position} has shape {plane.shape} but class 0 has {codes.shape}; "
                "one file per class means one labelling of one image, so they must be the "
                "same size"
            )
        above = plane > threshold
        codes[above] |= 1 << position
        hit |= above
    return codes, float(1.0 - hit.mean())


def codes_to_onehot(codes: np.ndarray, n_class: int) -> np.ndarray:
    """Bit-coded class membership to a channels-last one-hot array.

    The third of three expansions, and the only one that is arithmetic rather than a table.
    `np.eye(n_class)[classes]` is the exclusive case; `region_vocabulary`'s table is the
    nested one, one row per label value; this would need a row per *combination* - two to
    the power of the classes - so the bits are read directly instead.
    """
    highest = int(np.asarray(codes).max(initial=0))
    if highest >> n_class:
        raise MaskError(
            f"a voxel is coded as belonging to class {highest.bit_length() - 1} but only "
            f"{n_class} classes are defined"
        )
    bits = np.arange(n_class, dtype=np.int64)
    return ((np.asarray(codes)[..., None] >> bits) & 1).astype(np.float32)


def colours_to_classes(
    mask: np.ndarray, colormap: list[tuple[int, int, int]], *, tolerance: int = 0
) -> tuple[np.ndarray, float]:
    """Map an RGB mask to class indices, dropping the channel axis.

    Anything matching no colour becomes class 0. `unmatched_fraction` exists so a caller
    can notice when that is happening to most of the image, which usually means the
    colormap does not describe this dataset.
    """
    if mask.ndim < 2:
        raise MaskError(f"a mask needs at least 2 dimensions, got shape {mask.shape}")
    if mask.shape[-1] < 3:
        raise MaskError(f"expected an RGB mask with 3 channels last, got shape {mask.shape}")
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


def labels_to_classes(
    mask: np.ndarray, labels: list[int], *, tolerance: float = 0.5
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


def region_vocabulary(sets: Sequence[Sequence[int]]) -> tuple[list[int], np.ndarray]:
    """The label-map values nested regions mention, and which regions each one belongs to.

    Returns `(values, table)`. `values` is every value any region lists, sorted, and it is
    the vocabulary a label map is read into. `table` has one row per vocabulary entry plus
    one at the front, and one column per region, so `table[index]` is the set of regions a
    voxel reading `values[index - 1]` belongs to.

    **Row 0 is zeros, and it is the whole reason this is one function rather than two.** It
    means "a value no region mentions", and such a voxel has to be absent from every channel.
    `labels_to_classes` sends an unmatched voxel to index 0 because index 0 is a class there -
    the background a colormap puts first - and reusing that unchanged here would make
    background a member of whichever region holds the lowest value. On BraTS, where
    `TC = {1, 4}` and `WT = {1, 2, 4}`, it would put every background voxel in both.

    Returning the pair together is deliberate: a table built against one vocabulary and
    indexed with another is wrong in a way that still produces a plausible target.

        >>> values, table = region_vocabulary([[4], [1, 4], [1, 2, 4]])
        >>> values
        [1, 2, 4]
        >>> table.tolist()
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    """
    if not sets:
        raise MaskError("no regions to build a vocabulary from")
    values = sorted({int(value) for one in sets for value in one})
    table = np.zeros((len(values) + 1, len(sets)), dtype=np.float32)
    for column, one in enumerate(sets):
        for value in one:
            table[values.index(int(value)) + 1, column] = 1.0
    return values, table


def labels_to_regions(
    mask: np.ndarray, values: Sequence[int], *, tolerance: float = 0.5
) -> tuple[np.ndarray, float]:
    """Map a label map to indices into a region vocabulary, dropping the channel axis.

    The counterpart of `labels_to_classes` for nested regions, and it differs in one place:
    **index 0 is reserved for a value no region mentions**, so the listed values start at 1.
    The returned array is still one integer per voxel, which is what lets tiling and
    augmentation stay untouched - the expansion to one binary channel per region happens in
    the last step before the loss, through the table `region_vocabulary` returns.

    Compared with a tolerance for the reason `labels_to_classes` is: the values arrive as
    floats, and a label map that has been resampled or merely round-tripped through float32
    will not satisfy `== 2` reliably.
    """
    if mask.ndim < 2:
        raise MaskError(f"a mask needs at least 2 dimensions, got shape {mask.shape}")
    actual = mask[..., 0] if mask.shape[-1] == 1 else mask
    if actual.shape != mask.shape[:-1]:
        raise MaskError(
            f"a label map must have one channel, got shape {mask.shape}. A mask with "
            "several channels is a picture - use a colormap for it."
        )

    indices = np.zeros(actual.shape, dtype=np.int64)
    matched = np.zeros(actual.shape, dtype=bool)
    for position, value in enumerate(values, start=1):
        hit = np.abs(actual - float(value)) <= tolerance
        indices[hit] = position
        matched |= hit
    return indices, float(1.0 - matched.mean())


def signed_distance(onehot: np.ndarray, spacing: tuple[float, ...] | None = None) -> np.ndarray:
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
        raise MaskError(f"expected a one-hot mask with a class axis last; got shape {array.shape}")
    out = np.zeros(array.shape, dtype=np.float32)
    for index in range(array.shape[-1]):
        inside = array[..., index] > 0.5
        if not inside.any() or inside.all():
            continue
        out[..., index] = distance_transform_edt(
            ~inside, sampling=spacing
        ) - distance_transform_edt(inside, sampling=spacing)
    return out


def classes_to_onehot(classes: np.ndarray, n_class: int) -> np.ndarray:
    """Class indices to a channels-last one-hot array."""
    highest = int(classes.max(initial=0))
    if highest >= n_class:
        raise MaskError(f"found class index {highest} but only {n_class} classes are defined")
    return np.eye(n_class, dtype=np.float32)[classes]


def onehot_to_foreground(onehot: np.ndarray, *, multilabel: bool = False) -> np.ndarray:
    """Which voxels the mask assigns to something, as a channels-last 0/1 array.

    One place for a convention the package already depends on everywhere else: in the two
    exclusive namings **channel 0 is background**, because `colours_to_classes` and
    `labels_to_classes` both send an unmatched voxel to index 0 and a colormap puts the
    background first. With `regions` there is no background channel at all, so foreground
    is any channel being hot - `region_vocabulary`'s row of zeros is what makes an
    unmentioned value land in none of them.

    This exists for the regression losses, which need to know where a continuous target is
    defined. It is deliberately *not* read off the values of that target: a flow field is
    zero at the centre of every object, so the non-zero voxels are the foreground minus
    exactly the seeds.

    The limit worth stating: with `labels` that do not list 0 the first entry is a real
    class rather than background, and then this reads that class as background. Nothing in
    the spec forbids `labels: [1, 2]`, and nothing else in the package treats it any
    differently - index 0 doubles as "matched nothing" there, which is the older wart this
    one rides on. Measured and filed as `maju116/platypus#153`.
    """
    hot = np.asarray(onehot) >= 0.5
    foreground = hot.any(axis=-1) if multilabel else ~hot[..., 0]
    return foreground.astype(np.float32)[..., None]


def onehot_to_colours(onehot: np.ndarray, colormap: list[tuple[int, int, int]]) -> np.ndarray:
    """A predicted one-hot (or probability) array back to an RGB picture.

    Refuses an array where some voxel is confidently in two classes at once, rather than
    taking the `argmax` and writing one of them. **A colour map cannot hold an overlap**:
    one voxel gets one colour, so a voxel in both the tumour core and the whole tumour would
    be written as whichever channel happened to read higher, and the picture would be a
    plausible answer to a question nobody asked. Nested regions therefore have no colormap
    at all, and this is the function that says so instead of guessing.

    Detected from the array rather than from a flag, because that is what a caller holding
    only a prediction can answer. A softmax output never trips it: its channels sum to one,
    so at most one can reach half.
    """
    if onehot.shape[-1] != len(colormap):
        raise MaskError(
            f"prediction has {onehot.shape[-1]} channels but the colormap defines "
            f"{len(colormap)} classes"
        )
    hot = np.asarray(onehot) >= 0.5
    overlapping = int(hot.sum(axis=-1).max(initial=0))
    if overlapping > 1:
        raise MaskError(
            f"a voxel is in {overlapping} classes at once, and one voxel gets one colour - "
            "taking the strongest would write a plausible picture of something else. "
            "Overlapping classes are written one file per class, not as a colour map."
        )
    return np.asarray(colormap, dtype=np.uint8)[onehot.argmax(axis=-1)]


def class_filename(name: str) -> str:
    """A region's name as a filename fragment, which is not the same string.

    `regions` names are written for a reader - `whole tumour`, `enhancing` - and a reader is
    not a filesystem. Anything outside `[A-Za-z0-9._-]` becomes `_`, runs collapse, and the
    name is otherwise left alone: lowercasing it would make two platforms disagree about
    whether `Core` and `core` are one file, and this package already has a case-sensitivity
    story it does not need a second one of.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not cleaned:
        raise MaskError(
            f"the region name {name!r} has no character a filename can hold, so there is "
            "nothing to call its file. Name it in letters, digits, dot, dash or underscore."
        )
    return cleaned


def save_class_masks(
    prediction: np.ndarray,
    directory: str | Path,
    *,
    classes: Sequence[str],
    cases: Sequence[str] | None = None,
    threshold: float = 0.5,
    suffix: str = "",
) -> list[Path]:
    """One binary picture per class, which is how an overlapping prediction reaches disk.

    `onehot_to_colours` refuses a prediction where a voxel is in two classes at once, and
    its refusal says overlapping classes are written one file per class. This is that file
    per class. `regions` gives a channel per named region and a sigmoid per channel, so the
    channels neither compete nor sum to one, and the only lossless thing a picture can hold
    is one channel.

    Each file is `<case><suffix>_<region>.png`, mode `L`, 0 or 255 - a mask anything opens
    and anything thresholds back, rather than a colour that has to be looked up. The class
    name goes through `class_filename`, so `whole tumour` is `whole_tumour` on disk.

    The argument is `classes` rather than `names` because the R package's `save_masks` and
    `save_volumes` already use `names` for the **files** they write, and a sibling writer
    where the same word means the classes instead is the quiet kind of mistake: a call
    copied from one to the other would pass filenames as class names, the counts would
    match whenever they happened to, and nothing would refuse it.

    What this does **not** do is decide for you. A channel where nothing reaches `threshold`
    is written as an empty picture rather than skipped: a missing file reads as "this run did
    not produce that class", and an empty one reads as "it did and the answer was nothing",
    which are different findings and only one of them is true.

    Args:
        prediction: Channels-last probabilities or a one-hot array, either `(h, w, c)` for
            one image or `(n, h, w, c)` for a stack. A **volume** cannot come here: a PNG
            carries no voxel spacing and no orientation, so a volume mask written as
            pictures cannot be laid back over the scan it was computed from. Only rank 5 is
            refused, because a single volume `(d, h, w, c)` and a stack of images have the
            same rank and nothing in the array can tell them apart.
        directory: Where to write. Created, with parents, if it does not exist.
        classes: One name per channel, in channel order - `[region.name for region in
            spec.data.regions]`. Channel order is the order the regions were written in,
            which is also the order every per-class score is reported in.
        cases: One name per image, usually the source filenames' stems. Defaults to
            `mask_0001` and so on.
        threshold: A channel is hot where it reaches this. 0.5 is what every metric in the
            package thresholds a multilabel channel at, so a different number here writes a
            picture that disagrees with the score printed beside it.
        suffix: Added before the class, so two models can write into one directory.

    Returns:
        The paths written, image by image and within an image channel by channel.
    """
    array = np.asarray(prediction)
    if array.ndim >= 5:
        raise MaskError(
            f"this is a stack of volumes ({' x '.join(str(size) for size in array.shape)}), "
            "and a PNG carries neither voxel spacing nor orientation - a volume written as "
            "pictures cannot be laid back over the scan it came from. Write each channel as "
            "NIfTI with the geometry of its source instead."
        )
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise MaskError(
            f"expected a channels-last prediction, got shape "
            f"{' x '.join(str(size) for size in np.asarray(prediction).shape)}"
        )

    channels = array.shape[-1]
    if len(classes) != channels:
        raise MaskError(
            f"the prediction has {channels} channels and {len(classes)} class names were "
            "given. One name per channel, in the order the regions were written."
        )

    files = [class_filename(str(name)) for name in classes]
    if len(set(files)) != len(files):
        # Two names the filesystem cannot keep apart would be one file holding the second
        # channel, with the first silently gone and the count of written paths still right.
        clash = sorted({name for name in files if files.count(name) > 1})
        raise MaskError(
            f"two region names become the same filename: {', '.join(clash)}. They differ as "
            "prose and not as paths, so one class would overwrite the other."
        )

    count = len(array)
    if cases is None:
        cases = [f"mask_{index + 1:04d}" for index in range(count)]
    elif len(cases) != count:
        raise MaskError(f"there are {count} predictions and {len(cases)} case names were given")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    hot = array >= threshold

    written: list[Path] = []
    for index, case in enumerate(cases):
        for channel, name in enumerate(files):
            path = target / f"{case}{suffix}_{name}.png"
            picture = np.where(hot[index, ..., channel], 255, 0).astype(np.uint8)
            Image.fromarray(picture, mode="L").save(path)
            written.append(path)
    return written
