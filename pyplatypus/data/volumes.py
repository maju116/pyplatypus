"""Reading volumes, and the two things that make two scans comparable.

Opening a NIfTI is one line and, as with DICOM, not the point. Two corrections have to
happen before an array means anything:

**Orientation.** A NIfTI stores an affine, not an array in a known order. The same anatomy
can arrive as (sagittal, coronal, axial) or any of the other 47 permutations and flips, and
nothing in the array says which. Read two datasets naively and one of them has the patient's
left on the right; train on both and the model learns to distinguish the source rather than
the anatomy. Every volume here is reoriented to the closest canonical (RAS) orientation, so
axis order and direction mean the same thing for every file.

**Intensity.** CT volumes are in Hounsfield units, where air is -1000 and water is 0 on
every scanner ever built, and a network needs 0-1. Scaling each volume by its own extremes
throws that away and lets one bright voxel - a hip implant, a marker - rescale everything
else. A named window does not: `window="lung"` maps the same numbers to the same values in
every scan, which is the whole point of the units existing.

Spacing is carried along rather than applied. Resampling every volume to isotropic
millimetres is a real thing to want and a decision with consequences - it changes the voxel
grid the model sees - so it belongs in the pipeline as an explicit choice, not hidden inside
a reader. What the reader must not do is silently lose the information needed to make it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.common import WINDOWS

VOLUME_SUFFIXES = (".nii", ".nii.gz", ".mgz", ".mgh", ".img", ".hdr")


class VolumeError(PlatypusError):
    kind = "volume_error"


def looks_like_volume(path: str | Path) -> bool:
    """Whether this path is a volume file, by name.

    Unlike DICOM, which is detected by content because exports so often carry no useful
    extension, NIfTI is reliably named - and reading a header to find out would mean opening
    every PNG in a dataset to ask whether it is a brain.
    """
    name = str(path).lower()
    return any(name.endswith(suffix) for suffix in VOLUME_SUFFIXES)


def read_volume(
    path: str | Path,
    *,
    window: str | tuple[float, float] = "auto",
    size: tuple[int, ...] | None = None,
    nearest: bool = False,
    channels: int = 1,
) -> np.ndarray:
    """Read a volume as a channels-last array, reoriented to canonical and scaled to 0-1.

    `nearest` must be used for label maps: interpolating labels invents values that belong
    to no class, and 1.5 between liver and tumour is neither.
    """
    image = _load_canonical(path)
    values = np.asanyarray(image.dataobj, dtype=np.float32)

    if values.ndim == 3:
        values = values[..., None]
    elif values.ndim == 4:
        pass  # already has a trailing axis: time, or channels
    else:
        raise VolumeError(
            f"'{path}' has shape {values.shape}; a volume must have 3 spatial dimensions, "
            "optionally followed by one more axis"
        )

    values = _rescale(values, window, nearest=nearest)

    if channels != values.shape[-1]:
        if values.shape[-1] == 1:
            values = np.repeat(values, channels, axis=-1)
        else:
            raise VolumeError(
                f"'{path}' has {values.shape[-1]} channels but {channels} were asked for"
            )

    if size is not None:
        values = resize_volume(values, size, nearest=nearest)
    return values


def volume_spacing(path: str | Path) -> tuple[float, float, float]:
    """Millimetres per voxel, in the canonical axis order the reader returns.

    Needed by anything that resamples, and by anything reporting a volume in millilitres
    rather than in voxels - which is the number a clinician asked for.
    """
    zooms = _load_canonical(path).header.get_zooms()[:3]
    return tuple(float(z) for z in zooms)


def _load_canonical(path: str | Path):
    """Open a volume and put it in canonical orientation, in one place.

    The exceptions are named rather than caught blindly: a missing or unreadable file is an
    OSError, a file that is not a volume is nibabel's own ImageFileError, and a truncated
    one usually surfaces as EOFError out of gzip or a ValueError out of the header parser.
    Anything else is a bug worth seeing rather than rewording.
    """
    import nibabel as nib

    try:
        return nib.as_closest_canonical(nib.load(str(path)))
    except (OSError, EOFError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise VolumeError(f"could not read '{path}': {error}") from None


def volume_shape(path: str | Path) -> tuple[int, int, int]:
    """The canonical shape of a volume, from its header alone."""
    return tuple(int(size) for size in _load_canonical(path).shape[:3])


def _rescale(values: np.ndarray, window: str | tuple[float, float], *, nearest: bool) -> np.ndarray:
    """Map values to 0-1 through a fixed window, or leave a label map alone.

    A label map must keep its integers: 0, 1, 2 are classes, not intensities, and dividing
    them by their maximum turns class 1 of 2 into 0.5. `nearest` is how the caller says
    "these are labels", the same flag that already says "do not interpolate".
    """
    if nearest:
        return values

    if isinstance(window, str):
        if window in ("auto", "full"):
            # NIfTI stores no display window, so there is nothing to honour - unlike DICOM,
            # where "auto" means the window the scanner recorded. Falling back to the range
            # present is stated rather than silent, because it is the one case where a
            # bright outlier can still rescale the rest.
            low, high = float(values.min()), float(values.max())
        elif window in WINDOWS:
            # The same (centre, width) table the DICOM reader uses, so 'lung' means one
            # thing across the package rather than two things that nearly agree.
            centre, width = WINDOWS[window]
            low, high = centre - width / 2, centre + width / 2
        else:
            raise VolumeError(
                f"unknown window '{window}'; use 'auto', 'full', a (centre, width) pair, "
                f"or one of: {', '.join(sorted(WINDOWS))}"
            )
    else:
        centre, width = float(window[0]), float(window[1])
        if width <= 0:
            raise VolumeError(f"window width must be positive, got {width}")
        low, high = centre - width / 2, centre + width / 2

    if high <= low:
        return np.zeros_like(values)
    clipped = np.clip(values, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def resize_volume(array: np.ndarray, size: tuple[int, ...], *, nearest: bool = False) -> np.ndarray:
    """Resample a channels-last volume to `size`.

    Through torch rather than scipy: torch is already a dependency, does trilinear and
    nearest on the GPU or the processor, and one library fewer is one version conflict
    fewer for someone installing this into an environment they did not choose.
    """
    import torch

    if len(size) != 3:
        raise VolumeError(f"a volume needs three sizes, got {tuple(size)}")
    if array.ndim != 4:
        raise VolumeError(f"expected a channels-last volume (d, h, w, c), got shape {array.shape}")
    if tuple(array.shape[:3]) == tuple(size):
        return array

    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor.permute(3, 0, 1, 2)[None]  # (1, c, d, h, w)
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    resized = torch.nn.functional.interpolate(
        tensor, size=tuple(int(s) for s in size), mode=mode, **kwargs
    )
    return resized[0].permute(1, 2, 3, 0).numpy()


def resample_to_spacing(
    array: np.ndarray, source_spacing, target_spacing, *, nearest: bool = False
) -> np.ndarray:
    """Resample a channels-last volume from one voxel size to another.

    Why this is not the same as resizing to a fixed shape, which is what the pipeline does
    without it: a chest CT of 40 slices and one of 200 squeezed into the same box come out
    with the same number of voxels and *different physical scale*, so the same organ is twice
    the size in one and half in the other. The network has no way to know. Resampling fixes
    the millimetres per voxel instead and lets the shape vary, which is the quantity anatomy
    is actually measured in.
    """
    source = tuple(float(s) for s in source_spacing)
    target = tuple(float(s) for s in target_spacing)
    if len(source) != 3 or len(target) != 3:
        raise VolumeError(f"spacing is three numbers per side; got {source} and {target}")
    if any(s <= 0 for s in source + target):
        raise VolumeError(f"spacing must be positive; got {source} and {target}")
    if array.ndim != 4:
        raise VolumeError(f"expected a channels-last volume (d, h, w, c), got shape {array.shape}")

    shape = tuple(
        max(1, round(size * have / want))
        for size, have, want in zip(array.shape[:3], source, target, strict=True)
    )
    return resize_volume(array, shape, nearest=nearest)


def crop_or_pad(array: np.ndarray, size: tuple[int, ...], *, pad_value: float = 0.0) -> np.ndarray:
    """Centre-crop or centre-pad a channels-last array to exactly `size`.

    The other half of resampling. Once voxels are a fixed physical size the shape varies from
    case to case, and a network needs one shape - so the volume is cut or padded rather than
    stretched. Stretching would undo the resampling it just went through.

    Centred on purpose: a scan is framed around its subject, so what falls off the edge is
    background on both sides, and what gets added is background too.
    """
    if array.ndim != len(size) + 1:
        raise VolumeError(
            f"cannot fit shape {array.shape} to {tuple(size)}: the array needs "
            f"{len(size)} spatial axes plus channels"
        )

    result = array
    for axis, want in enumerate(int(s) for s in size):
        have = result.shape[axis]
        if have == want:
            continue
        if have > want:
            start = (have - want) // 2
            index = [slice(None)] * result.ndim
            index[axis] = slice(start, start + want)
            result = result[tuple(index)]
        else:
            before = (want - have) // 2
            after = want - have - before
            widths = [(0, 0)] * result.ndim
            widths[axis] = (before, after)
            result = np.pad(result, widths, mode="constant", constant_values=pad_value)
    return result


def _volume_cases(count: int, cases: Sequence[str] | None) -> list[str]:
    """Per-file names, defaulted and checked for collisions.

    The collision is the reason this is its own function rather than three lines twice.
    Volume datasets keep one directory per case with the same filename inside each -
    `case_01/images/ct.nii.gz` - so names taken from the sources collide, every mask is
    written to one path, and the count of written paths is still right. Losing five masks
    of six while reporting success is the worst outcome available here.
    """
    if cases is None:
        return [f"mask_{index + 1:04d}" for index in range(count)]
    names = [str(name) for name in cases]
    if len(names) != count:
        raise VolumeError(f"there are {count} predictions and {len(names)} case names were given")
    if len(set(names)) != len(names):
        clash = sorted({name for name in names if names.count(name) > 1})
        raise VolumeError(
            f"these case names repeat: {', '.join(clash[:3])}"
            f"{', ...' if len(clash) > 3 else ''}. Each would be written to the same path, "
            "so every mask but the last would be lost with nothing reporting it. Datasets "
            "that keep one directory per case usually hold the same filename in each, so "
            "the stems are not distinct - pass the keys instead."
        )
    return names


def _as_volume_list(
    prediction: np.ndarray | Sequence[np.ndarray], *, labels_allowed: bool = False
) -> tuple[list[np.ndarray], bool]:
    """One volume per case, and whether they are label maps or channels.

    `predict(space="source")` hands back a list because sources differ in size, and that is
    the input this writer is for. A single volume is accepted as a list of one, and a
    stacked array is unstacked.

    **A label map is accepted where one can be told apart from a prediction**, which is what
    `labels_allowed` is for: `save_volumes` takes either, because the R package's `predict()`
    returns class indices where this one returns probabilities, and expanding a label map
    into one-hot to collapse it again costs a gigabyte of float for a 512-cubed volume.
    `save_class_volumes` takes channels only - a binary file per class is a question about
    channels, and a label map has none.

    The rank decides, and **one case is genuinely ambiguous and is refused rather than
    guessed**: a stack of label maps `(n, d, h, w)` and one volume's channels `(d, h, w, c)`
    are both rank 4 and nothing in the array distinguishes them. `save_class_masks` names
    the same ambiguity for pictures and the R package resolves it the same way, so a caller
    meeting it here meets one rule rather than a third.
    """
    listed = isinstance(prediction, (list, tuple))
    if listed:
        volumes = [np.asarray(one) for one in prediction]
    else:
        array = np.asarray(prediction)
        volumes = list(array) if array.ndim == 5 else [array]

    ranks = {volume.ndim for volume in volumes}
    if len(ranks) != 1:
        raise VolumeError(
            "the volumes have different ranks "
            f"({', '.join(str(rank) for rank in sorted(ranks))}), so they cannot all be "
            "label maps or all be channels"
        )
    rank = ranks.pop()

    if rank == 4:
        if labels_allowed and not listed and np.asarray(prediction).ndim == 4:
            raise VolumeError(
                "this is rank 4 and could be either a stack of label maps "
                f"({' x '.join(str(size) for size in np.asarray(prediction).shape)} as "
                "`n x d x h x w`) or one volume's channels (as `d x h x w x c`), and "
                "nothing in the array says which. Pass a list - which is what "
                "`predict(space='source')` returns - or keep the volume axis."
            )
        return volumes, False
    if rank == 3 and labels_allowed:
        return volumes, True

    expected = (
        "One `(d, h, w, c)` array per case, which is what `predict(space='source')` "
        "returns, or a `(d, h, w)` map of class indices."
        if labels_allowed
        else "One `(d, h, w, c)` array per case, which is what `predict(space='source')` returns."
    )
    raise VolumeError(
        f"expected channels-last volumes, got shape "
        f"{' x '.join(str(size) for size in volumes[0].shape)}. {expected}"
    )


def _reference_affine(path: str | Path, shape: tuple[int, ...]):
    """The canonical image a mask is written against, with its geometry checked.

    Through `_load_canonical`, which is the whole correctness of this pair of functions
    rather than a convenience. `read_volume` reorients every volume it reads to the closest
    canonical (RAS) axes, so a prediction is in canonical space; writing it with the
    **stored** affine of a file whose axes are ordered differently produces a mask that
    opens without complaint and is reflected or transposed with respect to its scan. The
    reference has to be read the way the image was read, and sharing the function is what
    guarantees it stays that way.
    """
    image = _load_canonical(path)
    if tuple(image.shape[:3]) != tuple(shape):
        raise VolumeError(
            f"the mask is {' x '.join(str(size) for size in shape)} and '{path}' is "
            f"{' x '.join(str(size) for size in image.shape[:3])}. A mask written with the "
            "wrong geometry lands in the wrong place in the scan, which is worse than "
            "failing - predict with `space='source'` to get the grid the file is on."
        )
    return image


def _write_labels(labels: np.ndarray, path: Path, image) -> Path:
    """One integer-labelled NIfTI, carrying the reference's canonical affine."""
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    written = nib.Nifti1Image(labels.astype(np.int16), image.affine, dtype=np.int16)
    nib.save(written, str(path))
    return path


def save_volumes(
    prediction: np.ndarray | Sequence[np.ndarray],
    directory: str | Path,
    *,
    reference: Sequence[str | Path],
    cases: Sequence[str] | None = None,
    suffix: str = "",
) -> list[Path]:
    """A predicted volume as NIfTI, written onto the geometry of the scan it came from.

    This is what `save_class_masks`'s own refusal tells a reader to do, and until now there
    was nowhere in this package to do it: a PNG carries neither voxel spacing nor
    orientation, so a volume written as pictures cannot be laid back over its scan. A mask
    array on its own has the same problem. **The affine is the point** - every viewer, every
    registration tool and every volume in millilitres reads position and spacing out of it,
    and a file carrying the wrong one is either silently misplaced or rejected.

    Written as integer labels, `int16`, because that is what a volume format holds and what
    the next tool expects to read back. `reference` is required rather than optional for the
    same reason: there is no sensible affine to invent.

    Refuses an overlapping prediction instead of taking the `argmax`, exactly as
    `onehot_to_colours` does and for the same reason one level along: a label map holds one
    integer per voxel, so a voxel in both the tumour core and the whole tumour would be
    written as whichever channel read higher. Those are `regions`, and
    `save_class_volumes` is where they go. Detected from the array rather than from a flag,
    so a caller holding only a prediction can answer it, and a softmax never trips it.

    Args:
        prediction: Channels-last probabilities. A list of `(d, h, w, c)` arrays, which is
            what `predict(space="source")` returns, or one such array, or a stacked
            `(n, d, h, w, c)`.
        directory: Where to write. Created, with parents, if it does not exist.
        reference: The scan each mask was predicted from, in the same order. Its canonical
            affine is what the mask is written with.
        cases: One name per volume, without extension - usually `split`'s keys. Defaults to
            `mask_0001` and so on. Names that repeat are refused rather than overwritten.
        suffix: Added before the extension, so two models can write into one directory.

    Returns:
        The paths written, in the order the predictions were given.
    """
    volumes, already_labels = _as_volume_list(prediction, labels_allowed=True)
    sources = [str(path) for path in reference]
    if len(sources) != len(volumes):
        raise VolumeError(
            f"there are {len(volumes)} masks and {len(sources)} reference scans. Each mask "
            "is written with the geometry of the scan it came from, so they have to line up."
        )
    names = _volume_cases(len(volumes), cases)

    written: list[Path] = []
    for volume, source, case in zip(volumes, sources, names, strict=True):
        if already_labels:
            # No overlap check, and that is not a hole. The refusal below protects the
            # *collapse*: it exists because an argmax over channels silently picks one of
            # the classes a voxel is in. An integer per voxel cannot express an overlap in
            # the first place, so there is nothing here to detect and nothing being lost.
            image = _reference_affine(source, volume.shape[:3])
            written.append(_write_labels(volume, Path(directory) / f"{case}{suffix}.nii.gz", image))
            continue
        hot = volume >= 0.5
        overlapping = int(hot.sum(axis=-1).max(initial=0))
        if overlapping > 1:
            raise VolumeError(
                f"a voxel is in {overlapping} classes at once, and a label map holds one "
                "integer per voxel - taking the strongest would write a plausible volume of "
                "something else. These are overlapping classes (`regions`), so write them "
                "one file per class with `save_class_volumes`."
            )
        image = _reference_affine(source, volume.shape[:3])
        written.append(
            _write_labels(volume.argmax(axis=-1), Path(directory) / f"{case}{suffix}.nii.gz", image)
        )
    return written


def save_class_volumes(
    prediction: np.ndarray | Sequence[np.ndarray],
    directory: str | Path,
    *,
    reference: Sequence[str | Path],
    classes: Sequence[str],
    cases: Sequence[str] | None = None,
    threshold: float = 0.5,
    suffix: str = "",
) -> list[Path]:
    """One binary volume per class, which is how an overlapping prediction reaches disk in 3D.

    `save_class_masks` is this function for pictures, and `regions` was motivated by BraTS,
    which stores its overlapping regions as volumes - so this is the writer the motivating
    dataset actually needs. Each file is `<case><suffix>_<class>.nii.gz`, 0 or 1 as `int16`,
    carrying the reference's canonical affine like everything else here.

    `classes` rather than `names`, and `cases` for the per-file half, which is the same
    split `save_class_masks` settled: in the R package `names` already means the files a
    writer writes, so a sibling where the same word meant the classes would accept a call
    copied from one to the other whenever the counts happened to match.

    A channel where nothing reaches `threshold` is written as an empty volume rather than
    skipped, for `save_class_masks`'s reason: a missing file reads as "this run did not
    produce that class" and an empty one as "it did and the answer was nothing", and only
    one of those is true.

    Args:
        prediction: Channels-last probabilities, as for `save_volumes`.
        directory: Where to write. Created, with parents, if it does not exist.
        reference: The scan each mask was predicted from, in the same order.
        classes: One name per channel, in channel order. Passed through `class_filename`,
            so `whole tumour` is `whole_tumour` on disk.
        cases: One name per volume. Defaults to `mask_0001` and so on.
        threshold: A channel is hot where it reaches this. 0.5 is what every metric in the
            package thresholds a multilabel channel at.
        suffix: Added before the class, so two models can write into one directory.

    Returns:
        The paths written, volume by volume and within a volume channel by channel.
    """
    from pyplatypus.data.masks import class_filename

    volumes, _ = _as_volume_list(prediction)
    sources = [str(path) for path in reference]
    if len(sources) != len(volumes):
        raise VolumeError(
            f"there are {len(volumes)} masks and {len(sources)} reference scans. Each mask "
            "is written with the geometry of the scan it came from, so they have to line up."
        )
    names = _volume_cases(len(volumes), cases)

    files = [class_filename(str(name)) for name in classes]
    if len(set(files)) != len(files):
        clash = sorted({name for name in files if files.count(name) > 1})
        raise VolumeError(
            f"two class names become the same filename: {', '.join(clash)}. They differ as "
            "prose and not as paths, so one class would overwrite the other."
        )

    written: list[Path] = []
    for volume, source, case in zip(volumes, sources, names, strict=True):
        if volume.shape[-1] != len(files):
            raise VolumeError(
                f"the prediction has {volume.shape[-1]} channels and {len(files)} class "
                "names were given. One name per channel, in the order the regions were "
                "written."
            )
        image = _reference_affine(source, volume.shape[:3])
        for channel, name in enumerate(files):
            written.append(
                _write_labels(
                    (volume[..., channel] >= threshold).astype(np.int16),
                    Path(directory) / f"{case}{suffix}_{name}.nii.gz",
                    image,
                )
            )
    return written
