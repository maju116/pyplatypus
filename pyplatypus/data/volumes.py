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


def read_volume(path: str | Path, *, window: str | tuple[float, float] = "auto",
                size: tuple[int, ...] | None = None, nearest: bool = False,
                channels: int = 1) -> np.ndarray:
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


def _rescale(values: np.ndarray, window: str | tuple[float, float], *, nearest: bool
             ) -> np.ndarray:
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


def resize_volume(array: np.ndarray, size: tuple[int, ...], *, nearest: bool = False
                  ) -> np.ndarray:
    """Resample a channels-last volume to `size`.

    Through torch rather than scipy: torch is already a dependency, does trilinear and
    nearest on the GPU or the processor, and one library fewer is one version conflict
    fewer for someone installing this into an environment they did not choose.
    """
    import torch

    if len(size) != 3:
        raise VolumeError(f"a volume needs three sizes, got {tuple(size)}")
    if array.ndim != 4:
        raise VolumeError(
            f"expected a channels-last volume (d, h, w, c), got shape {array.shape}"
        )
    if tuple(array.shape[:3]) == tuple(size):
        return array

    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor.permute(3, 0, 1, 2)[None]           # (1, c, d, h, w)
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    resized = torch.nn.functional.interpolate(tensor, size=tuple(int(s) for s in size),
                                              mode=mode, **kwargs)
    return resized[0].permute(1, 2, 3, 0).numpy()
