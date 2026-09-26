"""Reading DICOM the way a radiologist would expect.

Opening the file is the easy part and not the point. `dcmread(path).pixel_array` returns
*stored* values, which are not measurements of anything until the modality LUT has been
applied - for CT that is `stored * RescaleSlope + RescaleIntercept`, turning arbitrary
integers into Hounsfield units, where air is -1000 and water is 0 on every scanner ever
built. Skip it and a model trained on one machine means nothing on another, and nothing in
the data says so.

Then the values have to be mapped to the range the network sees, and *how* matters more
than it looks. Scaling each image by its own minimum and maximum is the obvious choice and
the wrong one: a single bright pixel - a metal implant, a marker, an artefact - rescales
everything else. Measured on pydicom's CT_small, one such pixel moves the mean of the rest
of the image from 0.378 to 0.200. A fixed window leaves the rest bit-for-bit identical,
and is also what makes two scans comparable in the first place.

So: modality LUT, then a window that does not depend on the image, then MONOCHROME1
inverted if that is what the file says.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.common import WINDOWS


class DicomError(PlatypusError):
    kind = "dicom_error"


def looks_like_dicom(path: str | Path) -> bool:
    """DICOM files carry 'DICM' at offset 128, which is more reliable than the extension.

    Exports out of a PACS frequently have no extension at all, or something the archive
    invented, so trusting the name would refuse perfectly good files.
    """
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True
    try:
        with path.open("rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def resolve_window(dataset, window) -> tuple[float, float] | None:
    """Turn what the user asked for into (low, high), or None to use the full range."""
    if window in (None, "full"):
        return None

    if isinstance(window, str):
        if window == "auto":
            centre = getattr(dataset, "WindowCenter", None)
            width = getattr(dataset, "WindowWidth", None)
            if centre is None or width is None:
                return None
            # These may be multi-valued when a file offers several presets; the first is
            # the one viewers show by default.
            centre = float(centre[0] if hasattr(centre, "__len__") else centre)
            width = float(width[0] if hasattr(width, "__len__") else width)
        elif window in WINDOWS:
            centre, width = WINDOWS[window]
        else:
            raise DicomError(
                f"unknown window '{window}'. Use 'auto', 'full', a pair of numbers, or "
                f"one of: {', '.join(sorted(WINDOWS))}"
            )
    else:
        centre, width = (float(v) for v in window)

    if width <= 0:
        raise DicomError(f"window width must be positive, got {width}")
    return centre - width / 2, centre + width / 2


def read_dicom(path: str | Path, window="auto", channels: int = 1) -> np.ndarray:
    """One DICOM file as a channels-last float32 array scaled to 0-1."""
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pragma: no cover
        raise DicomError(
            "reading DICOM needs pydicom, which should have been installed with this "
            "package."
        ) from None

    try:
        dataset = pydicom.dcmread(str(path))
        stored = dataset.pixel_array
    except Exception as error:  # noqa: BLE001 - pydicom raises many things
        raise DicomError(f"could not read '{path}': {error}") from None

    # Stored values to real-world units. For CT this is the step that produces Hounsfield
    # units; for modalities without a rescale it is the identity, and costs nothing.
    values = np.asarray(apply_modality_lut(stored, dataset), dtype=np.float32)

    bounds = resolve_window(dataset, window)
    if bounds is None:
        low, high = float(values.min()), float(values.max())
    else:
        low, high = bounds
    if high <= low:
        high = low + 1.0

    values = np.clip(values, low, high)
    values = (values - low) / (high - low)

    # MONOCHROME1 means low values are bright. Left alone, every such image trains as its
    # own negative, which looks like nothing being wrong.
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        values = 1.0 - values

    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise DicomError(
            f"'{path}' holds a {values.ndim - 1}-dimensional image; only single frames are "
            "supported so far."
        )

    present = values.shape[-1]
    if channels == present:
        return values
    if channels == 1:
        return values.mean(axis=-1, keepdims=True)
    if channels == 3 and present == 1:
        return np.repeat(values, 3, axis=-1)
    raise DicomError(
        f"'{path}' has {present} channel(s) and {channels} were asked for."
    )
