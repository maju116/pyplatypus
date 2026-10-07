"""Assembling a directory of DICOM slices into one volume.

This is how data leaves a hospital: a folder of files, one per slice, named by whatever the
archive felt like. Turning it into a volume is four decisions, and getting any of them wrong
produces a volume that trains without complaint and means something other than it claims.

**Order.** Not the filename, and not `InstanceNumber` either. Filenames sort
lexicographically, so slice 10 lands before slice 2; `InstanceNumber` is *usually* right and
is not guaranteed to be, because it need only be unique within a series and some scanners
number by acquisition rather than by position. The position is in the file:
`ImagePositionPatient` projected onto the slice normal, which is the cross product of the two
direction cosines in `ImageOrientationPatient`. That is a physical distance along the axis the
slices advance in, and it cannot disagree with the anatomy.

**One series at a time.** A folder often holds several - a scout, several reconstructions, a
different phase - and `SeriesInstanceUID` is what separates them. Stacking two series into one
array interleaves two anatomies at two resolutions, which looks like a volume and is not.

**One window for the whole series.** `read_dicom` honours the window recorded in a file, and
different slices of one series can record different ones. Applied slice by slice, the same
tissue would be a different brightness on adjacent slices - a gradient the scanner never
measured. The window is resolved once, from the first slice, and then applied as fixed bounds
to all of them.

**Orientation.** The series carries a patient-space affine as surely as a NIfTI does, so it
goes through the same canonical reorientation. Without that, one CT read from its DICOM and
the same CT after conversion to NIfTI come out differently oriented, and nothing says so.

A gap in the positions is an error rather than a stacked array. A missing slice does not make
a volume slightly shorter, it makes everything past the gap sit in the wrong place, and a
model trained on it learns anatomy that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyplatypus.data.dicom import DicomError, looks_like_dicom, resolve_window
from pyplatypus.data.volumes import resize_volume

# Positions are in millimetres; scanners write them with a few decimals of noise.
POSITION_TOLERANCE = 1e-3
SPACING_TOLERANCE = 0.02  # 2% of the median gap, comfortably inside scanner jitter


@dataclass(frozen=True)
class Series:
    """One series, in the order the slices physically occur."""

    paths: tuple[Path, ...]
    affine: np.ndarray
    spacing: tuple[float, float, float]
    series_uid: str | None
    shape: tuple[int, int, int]
    sorted_by: str

    def __len__(self) -> int:
        return len(self.paths)


def looks_like_dicom_series(path: str | Path) -> bool:
    """A directory holding at least two DICOM files."""
    root = Path(path)
    if not root.is_dir():
        return False
    found = 0
    for entry in sorted(root.iterdir()):
        if entry.is_file() and looks_like_dicom(entry):
            found += 1
            if found >= 2:
                return True
    return False


def _collect(source: str | Path | list) -> list[Path]:
    if isinstance(source, (str, Path)):
        root = Path(source)
        if root.is_dir():
            paths = [p for p in sorted(root.iterdir()) if p.is_file() and looks_like_dicom(p)]
            if not paths:
                raise DicomError(f"'{root}' holds no DICOM files")
            return paths
        return [root]
    return [Path(p) for p in source]


def describe_series(source: str | Path | list) -> Series:
    """Sort a series and check it, without reading any pixels.

    Separate from reading because the checks are the expensive thinking and the pixels are
    the expensive bytes: a caller sorting a hundred cases wants to know which ones are
    broken before it reads a gigabyte.
    """
    import pydicom

    paths = _collect(source)
    headers = []
    for path in paths:
        try:
            headers.append((path, pydicom.dcmread(str(path), stop_before_pixels=True)))
        except Exception as error:  # noqa: BLE001 - pydicom raises many things
            raise DicomError(f"could not read '{path}': {error}") from None

    uids = {getattr(h, "SeriesInstanceUID", None) for _, h in headers}
    if len(uids) > 1:
        counts = {}
        for _, header in headers:
            uid = getattr(header, "SeriesInstanceUID", None)
            counts[uid] = counts.get(uid, 0) + 1
        listed = "; ".join(
            f"{uid}: {n} file(s)" for uid, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        raise DicomError(
            f"these files are {len(uids)} different series, not one: {listed}. Stacking "
            "them would interleave two anatomies. Separate them by SeriesInstanceUID first."
        )

    _check_geometry_agrees(headers)
    ordered, sorted_by = _order(headers)
    positions = [_position_along_normal(h) for _, h in ordered]
    slice_spacing = _check_gaps(ordered, positions, sorted_by)

    first = ordered[0][1]
    affine = _affine(first, ordered, slice_spacing)
    rows, columns = int(first.Rows), int(first.Columns)
    row_spacing, column_spacing = _pixel_spacing(first)

    return Series(
        paths=tuple(path for path, _ in ordered),
        affine=affine,
        spacing=(row_spacing, column_spacing, slice_spacing),
        series_uid=next(iter(uids)),
        shape=(rows, columns, len(ordered)),
        sorted_by=sorted_by,
    )


def read_dicom_series(
    source: str | Path | list,
    *,
    window="auto",
    channels: int = 1,
    size: tuple[int, ...] | None = None,
    nearest: bool = False,
) -> np.ndarray:
    """Read a series as one channels-last volume, canonical and scaled to 0-1."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    series = describe_series(source)

    # One window for the whole series, resolved from the first slice. Slice by slice, a
    # series whose files record different windows would come out with a brightness gradient
    # the scanner never measured.
    first = pydicom.dcmread(str(series.paths[0]))
    bounds = resolve_window(first, window)

    slices = []
    for path in series.paths:
        try:
            dataset = pydicom.dcmread(str(path))
            values = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)
        except Exception as error:  # noqa: BLE001
            raise DicomError(f"could not read '{path}': {error}") from None
        if values.ndim != 2:
            raise DicomError(
                f"'{path}' holds a {values.ndim}-dimensional frame; a series is assembled "
                "from single-frame slices."
            )
        slices.append(values)

    volume = np.stack(slices, axis=-1)  # (rows, columns, slices)

    if bounds is None:
        low, high = float(volume.min()), float(volume.max())
    else:
        low, high = bounds
    if high <= low:
        high = low + 1.0
    volume = np.clip(volume, low, high)
    volume = ((volume - low) / (high - low)).astype(np.float32)

    if getattr(first, "PhotometricInterpretation", "") == "MONOCHROME1":
        volume = 1.0 - volume

    volume = _to_canonical(volume, series.affine)
    volume = volume[..., None]

    if channels != 1:
        volume = np.repeat(volume, channels, axis=-1)
    if size is not None:
        volume = resize_volume(volume, size, nearest=nearest)
    return volume


def series_spacing(source: str | Path | list) -> tuple[float, float, float]:
    """Millimetres per voxel, in the canonical order the reader returns.

    Slice spacing is measured from the positions rather than read from `SliceThickness`,
    which describes how thick a slice is and not how far apart they sit - for an overlapping
    reconstruction those differ, and using thickness would put every slice past the first in
    the wrong place.
    """
    import nibabel as nib

    series = describe_series(source)
    image = nib.as_closest_canonical(
        nib.Nifti1Image(np.zeros(series.shape, dtype=np.float32), series.affine)
    )
    return tuple(float(z) for z in image.header.get_zooms()[:3])


def series_shape(source: str | Path | list) -> tuple[int, int, int]:
    """The canonical shape of a series, from headers alone.

    Not `Series.shape`, which is (rows, columns, slices) as the files store it: the reader
    reorients to canonical, so the shape that describes the returned array can be a
    permutation of that one.
    """
    import nibabel as nib

    series = describe_series(source)
    image = nib.as_closest_canonical(
        nib.Nifti1Image(np.zeros(series.shape, dtype=np.float32), series.affine)
    )
    return tuple(int(size) for size in image.shape[:3])


# --------------------------------------------------------------------- internals
def _pixel_spacing(header) -> tuple[float, float]:
    spacing = getattr(header, "PixelSpacing", None)
    if spacing is None:
        raise DicomError(
            "the slices carry no PixelSpacing, so the volume has no physical size. Any "
            "measurement taken from it would be in pixels pretending to be millimetres."
        )
    return float(spacing[0]), float(spacing[1])


def _orientation(header) -> np.ndarray | None:
    values = getattr(header, "ImageOrientationPatient", None)
    if values is None or len(values) != 6:
        return None
    return np.asarray([float(v) for v in values], dtype=np.float64)


def _normal(header) -> np.ndarray | None:
    orientation = _orientation(header)
    if orientation is None:
        return None
    return np.cross(orientation[:3], orientation[3:])


def _position_along_normal(header) -> float | None:
    position = getattr(header, "ImagePositionPatient", None)
    normal = _normal(header)
    if position is None or normal is None:
        return None
    return float(np.dot(np.asarray([float(v) for v in position]), normal))


def _check_geometry_agrees(headers) -> None:
    shapes = {(int(h.Rows), int(h.Columns)) for _, h in headers}
    if len(shapes) > 1:
        raise DicomError(
            f"the slices are not all the same size: {sorted(shapes)}. One series cannot "
            "change resolution partway through."
        )

    orientations = [_orientation(h) for _, h in headers]
    if all(o is not None for o in orientations):
        first = orientations[0]
        for path, orientation in zip([p for p, _ in headers], orientations, strict=True):
            if not np.allclose(orientation, first, atol=1e-4):
                raise DicomError(
                    f"'{path}' is oriented differently from the first slice. Slices at "
                    "different angles are not a volume - they are separate acquisitions."
                )

    spacings = {_pixel_spacing(h) for _, h in headers}
    if len(spacings) > 1:
        raise DicomError(
            f"the slices have different PixelSpacing: {sorted(spacings)}. The volume would "
            "have no single physical scale."
        )


def _order(headers):
    """Sorted slices, and what they were sorted by."""
    positions = [_position_along_normal(h) for _, h in headers]
    if all(p is not None for p in positions):
        order = sorted(range(len(headers)), key=lambda i: positions[i])
        return [headers[i] for i in order], "position"

    numbers = [getattr(h, "InstanceNumber", None) for _, h in headers]
    if all(n is not None for n in numbers):
        # A documented second best. InstanceNumber has to be unique within a series but is
        # not required to follow the anatomy, so this is used only when the geometry is
        # missing - and said out loud in `sorted_by` so a caller can notice.
        order = sorted(range(len(headers)), key=lambda i: int(numbers[i]))
        return [headers[i] for i in order], "instance_number"

    raise DicomError(
        "the slices carry neither ImagePositionPatient/ImageOrientationPatient nor "
        "InstanceNumber, so there is no way to tell what order they go in. Sorting by "
        "filename would be a guess, and a wrong order trains a model on anatomy that does "
        "not exist."
    )


def _check_gaps(ordered, positions, sorted_by) -> float:
    if sorted_by != "position":
        thickness = getattr(ordered[0][1], "SliceThickness", None)
        if thickness is None:
            raise DicomError(
                "without positions and without SliceThickness there is nothing to say how "
                "far apart the slices are."
            )
        return float(thickness)

    if len(ordered) < 2:
        thickness = getattr(ordered[0][1], "SliceThickness", None)
        return float(thickness) if thickness is not None else 1.0

    gaps = np.diff(np.asarray(positions, dtype=np.float64))
    duplicates = [i for i, gap in enumerate(gaps) if abs(gap) < POSITION_TOLERANCE]
    if duplicates:
        first = duplicates[0]
        raise DicomError(
            f"'{ordered[first][0].name}' and '{ordered[first + 1][0].name}' are at the same "
            "position. Two slices in one place means duplicated files, or several echoes or "
            "phases mixed together - not a volume."
        )

    median = float(np.median(gaps))
    tolerance = max(SPACING_TOLERANCE * abs(median), POSITION_TOLERANCE)
    off = [
        (ordered[i][0].name, float(gap))
        for i, gap in enumerate(gaps)
        if abs(gap - median) > tolerance
    ]
    if off:
        consequence = (
            "A volume stacked over a gap does not lose a slice, it puts everything past the "
            "gap in the wrong place - which no metric would show. Complete the series, or "
            "read the parts either side separately."
        )
        # With only two gaps the median sits between them and cannot say which one is wrong,
        # so listing them beats naming a culprit by arithmetic accident.
        if len(gaps) < 3:
            listed = ", ".join(
                f"{ordered[i][0].name} to {ordered[i + 1][0].name}: {float(gap):.3f} mm"
                for i, gap in enumerate(gaps)
            )
            raise DicomError(
                f"the slices are not evenly spaced ({listed}), and with this few slices "
                f"there is no way to tell which gap is the wrong one. {consequence}"
            )
        name, gap = off[0]
        raise DicomError(
            f"the slices are not evenly spaced: after '{name}' the gap is {gap:.3f} mm where "
            f"the rest are {median:.3f} mm. That is a missing slice. {consequence}"
        )
    return abs(median)


def _affine(first, ordered, slice_spacing: float) -> np.ndarray:
    """The patient-space affine for an array indexed (row, column, slice)."""
    orientation = _orientation(first)
    row_spacing, column_spacing = _pixel_spacing(first)
    origin = getattr(first, "ImagePositionPatient", None)

    if orientation is None or origin is None:
        # Sorted by InstanceNumber, so there is no patient-space geometry to honour: axis
        # aligned, spacing only. Enough to keep sizes and measurements honest, and the
        # canonical reorientation then has nothing to do.
        return np.diag([row_spacing, column_spacing, slice_spacing, 1.0])

    # ImageOrientationPatient is (direction of increasing column, direction of increasing
    # row): position = origin + row_dir * i * row_spacing + column_dir * j * column_spacing.
    column_dir = orientation[:3]
    row_dir = orientation[3:]

    if len(ordered) >= 2:
        start = np.asarray([float(v) for v in ordered[0][1].ImagePositionPatient])
        end = np.asarray([float(v) for v in ordered[-1][1].ImagePositionPatient])
        step = (end - start) / (len(ordered) - 1)
    else:
        step = np.cross(column_dir, row_dir) * slice_spacing

    affine = np.eye(4)
    affine[:3, 0] = row_dir * row_spacing
    affine[:3, 1] = column_dir * column_spacing
    affine[:3, 2] = step
    affine[:3, 3] = np.asarray([float(v) for v in origin])
    return affine


def _to_canonical(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Reorient to closest-canonical RAS, the same way a NIfTI is read.

    Through nibabel rather than by hand: it is already a dependency, and its orientation
    code is the reference implementation everything else in the field agrees with. Doing it
    here means a CT read from DICOM and the same CT converted to NIfTI arrive identically
    oriented, which is the only way a model can be trained on both.
    """
    import nibabel as nib

    image = nib.Nifti1Image(volume, affine)
    return np.asanyarray(nib.as_closest_canonical(image).dataobj, dtype=np.float32)
