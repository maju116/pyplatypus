"""Reading pixels, and cutting them up.

`tile` and `stitch` are inverses, and there is a round-trip test that says so. The old
package could only tile: an HD image went in as a grid of pieces and the predicted pieces
were never put back together, which defeated the purpose of tiling it.

Everything is rank-generic. A 2D grid and a 3D patch grid are the same operation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pyplatypus.errors import PlatypusError

# PIL talks in (width, height); every shape in this package is (height, width). Mixing
# them is the classic silent bug, so the conversion happens in exactly one place.
_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}

#: The mode a LABEL MAP is read in, and why it is not `_PIL_MODE[1]`. `"L"` is 8-bit, so
#: `convert("L")` flattens every value above 255 to 255 - and a label map's values are
#: class identities rather than brightness, so that merges two classes into one at an
#: unmatched share of exactly zero. `"I"` is 32-bit integer and keeps them. Not used for
#: images: `to_float` divides an ordinary picture by 255, and a one-channel image read
#: this way would stop being normalised. `pyplatypus#346`.
_LABEL_MODE = "I"

#: The modes whose stored numbers are **indices rather than brightness**, so `convert` would
#: answer a question about the palette or the bit depth instead of about the file. For these
#: a label map is read from the raw pixels and converted afterwards.
#:
#: `"P"` stores one palette index per pixel, which for a segmentation mask IS the class -
#: Pascal VOC's masks are stored this way and several annotation tools write them so.
#: `convert` expands the palette first and returns the luminance of each index's colour, so
#: indices 0, 1, 2 came back as 0, 38, 75 against a VOC-like palette. `"1"` is one bit per
#: pixel, and `convert` scales it to 0 and 255, so a mask storing 0 and 1 arrived as 0 and
#: 255 - which `labels: [0, 1]` then reads as one class present and one absent.
#:
#: `"F"` is 32-bit float, and `convert("I")` **truncates towards zero** rather than
#: rounding: a mask storing 0.9999 for class 1 read as 0, one class down, at an unmatched
#: share of exactly 0.0. `labels_to_classes` compares values to within half a unit for
#: precisely this reason and never got the chance, because the truncation happens first.
#: The docstring promising that "a 32-bit TIFF keeps the values it stores" covers a float
#: TIFF, and did not. `pyplatypus#361`.
#:
#: `_LABEL_MODE` alone was not enough for any of the three: `#346` widened the conversion
#: and these are reinterpreted *before* any width is involved. `pyplatypus#353`.
_STORED_VALUE_MODES = frozenset({"P", "1", "F"})


def _stored_values(handle: Image.Image) -> Image.Image:
    """The file's own numbers as an integer image, for a mode `convert` would reinterpret.

    Built rather than converted, so the resize and the `asarray` below are reached by one
    path for every mode. `np.asarray` on a `"P"` handle gives the indices, on a `"1"` handle
    gives booleans, and on an `"F"` handle gives float32; int32 is `_LABEL_MODE`'s own
    width, so nothing is clipped on the way through.

    **A float label map is ROUNDED, not truncated**, which is the whole of `pyplatypus#361`:
    a class written as 0.9999 by whatever produced the file is class 1, and `astype` alone
    would make it 0. Rounding here rather than reading it as float and leaving the half-unit
    tolerance to `labels_to_classes` keeps the dtype this function promises, and the two
    agree on every value either of them resolves.
    """
    array = np.asarray(handle)
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(array)
    return Image.fromarray(array.astype(np.int32), mode="I")


#: The formats `Image.open` is allowed to dispatch to, and the reason there is a list at all:
#: PIL identifies a file by its CONTENT and not by its name, so without this every decoder it
#: ships is reachable from any path a specification names. One of them, the EPS plugin, hands
#: the file to Ghostscript; others - FlashPix, JPEG 2000, MSP, WMF, XBM - are formats no
#: reader in this package has ever been pointed at. Naming the five that datasets actually
#: arrive in is the only repair for that which does not expire with a version: it is a
#: statement about what this package reads, rather than about which decoder bugs are open
#: today. PNG and JPEG are what every example and test here reads; TIFF, BMP and GIF are the
#: other mainstream raster formats a 2D set travels in, and refusing them would be a
#: narrowing nobody measured. Volumes, DICOM and DICOM series never reach PIL - they are
#: dispatched away above - so this list is about pictures only.
READABLE_FORMATS = ["PNG", "JPEG", "TIFF", "BMP", "GIF"]


class ImageError(PlatypusError):
    kind = "image_error"


def read_image(
    path: str | Path,
    *,
    channels: int = 3,
    size: tuple[int, ...] | None = None,
    nearest: bool = False,
    dicom_window="auto",
    label_map: bool = False,
) -> np.ndarray:
    """Read one image or volume as a channels-last array.

    The reader every other part of this package uses, which is the reason it is public: a
    picture drawn from a differently resized copy is a picture of something the model never
    saw, and the disagreements it shows may be the resizing rather than the model. Boxes
    come back from `predict` in the source image's own pixels, so drawing them needs the
    image at `size=None`.

    Scaled to 0-1 for anything read in real units (DICOM, NIfTI) and left at 0-255 for
    ordinary pictures, which `to_float` divides later. Both are drawable: the plotting
    functions decide by the range rather than by the dtype.

    DICOM is detected by content rather than by extension, because exports out of an
    archive frequently carry no extension or one the archive invented. A directory of
    slices is read as one volume, so a CSV may name a series directory in the column that
    names a file.

    Args:
        path: a file, or a directory holding one DICOM series.
        channels: how many channels to return - 1, 3 or 4 for ordinary pictures.
        size: resize to this shape, or `None` for the file's own. Give it in the order every
            shape in this package uses, `(height, width)` and `(height, width, depth)`.
        nearest: nearest-neighbour rather than interpolation. **Required for masks**:
            interpolating a mask invents colours that match no class, which then become
            background without a word.
        dicom_window: the window to apply to data in real units, `"auto"` to take the one
            the file records, or `(centre, width)`. Ignored for ordinary pictures.
        label_map: the file holds class identities rather than brightness, so read it at
            full integer width - a 16-bit PNG or a 32-bit TIFF keeps the values it stores
            instead of being flattened at 255. One channel only, because a label map has
            one. Ignored for DICOM and for volume formats, which never went through an
            8-bit conversion. **Not for images**: `to_float` divides an ordinary picture
            by 255.

    Returns:
        A channels-last array: `(height, width, channels)` for an image and
        `(height, width, depth, channels)` for a volume.

    Raises:
        ImageError: for a channel count an ordinary picture cannot have, or for
            `label_map` with more than one channel.

    No example: it reads a file, so there is nothing to show without one on disk. The data
    science bowl article draws what it returns.
    """
    from pyplatypus.data.dicom import looks_like_dicom, read_dicom
    from pyplatypus.data.dicom_series import looks_like_dicom_series, read_dicom_series
    from pyplatypus.data.volumes import looks_like_volume, read_volume

    # A directory of slices is one volume, which is how data leaves a hospital. Handled here
    # so a CSV can name a series directory in the same column that names a file.
    if looks_like_dicom_series(path):
        return read_dicom_series(
            path, window=dicom_window, channels=channels, size=size, nearest=nearest
        )

    # A volume asks for a different reader, not a different pipeline. Dispatching here keeps
    # every caller - the dataset, the plotting helpers, the R surface - reading "whatever is
    # at this path, the way training would", which is the only way a picture of a prediction
    # can be trusted to show what the model saw.
    if looks_like_volume(path):
        return read_volume(path, window=dicom_window, size=size, nearest=nearest, channels=channels)

    if looks_like_dicom(path):
        array = read_dicom(path, window=dicom_window, channels=channels)
        if size is not None:
            array = _resize_array(array, size, nearest=nearest)
        return array

    mode = _PIL_MODE.get(channels)
    if mode is None:
        raise ImageError(f"channels must be 1, 3 or 4 for ordinary images, got {channels}")
    if label_map:
        if channels != 1:
            raise ImageError(
                f"a label map has one channel, got channels={channels}. A mask with "
                "several channels is a picture - read it as one and use a colormap."
            )
        mode = _LABEL_MODE

    # int32 with the integer mode, for the reason the mode itself exists: asking numpy for
    # uint8 here would clip exactly what the conversion was widened to keep.
    dtype = np.int32 if label_map else np.uint8

    try:
        with Image.open(path, formats=READABLE_FORMATS) as handle:
            source = (
                _stored_values(handle)
                if label_map and handle.mode in _STORED_VALUE_MODES
                else handle
            )
            picture = source.convert(mode)
            if size is not None:
                if len(size) != 2:
                    raise ImageError(f"2D readers take a (height, width), got {tuple(size)}")
                resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
                picture = picture.resize((size[1], size[0]), resample=resample)
            array = np.asarray(picture, dtype=dtype)
    except UnidentifiedImageError:
        raise _not_a_readable_picture(path) from None
    except OSError as error:
        raise ImageError(f"could not read '{path}': {error}") from None

    if array.ndim == 2:
        array = array[..., None]
    return array


def spatial_shape(path: str | Path) -> tuple[int, ...]:
    """The size of what is at this path, without reading the pixels.

    Needed by anything that has to put a prediction back where it came from: the model works
    at its own size, and the answer belongs on the grid the data arrived on. Headers only,
    because asking this about a hundred cases should not cost a hundred gigabytes.

    For volumes the shape is the *canonical* one, matching what the reader returns - a NIfTI
    whose axes are stored in another order comes back permuted, and the shape has to agree with
    the array it describes rather than with the file.
    """
    from pyplatypus.data.dicom import looks_like_dicom
    from pyplatypus.data.dicom_series import looks_like_dicom_series, series_shape
    from pyplatypus.data.volumes import looks_like_volume, volume_shape

    if looks_like_dicom_series(path):
        return series_shape(path)
    if looks_like_volume(path):
        return volume_shape(path)
    if looks_like_dicom(path):
        import pydicom

        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
            return (int(header.Rows), int(header.Columns))
        except Exception as error:  # noqa: BLE001 - pydicom raises many things
            raise ImageError(f"could not read '{path}': {error}") from None

    try:
        with Image.open(path, formats=READABLE_FORMATS) as handle:
            width, height = handle.size
    except UnidentifiedImageError:
        raise _not_a_readable_picture(path) from None
    except OSError as error:
        raise ImageError(f"could not read '{path}': {error}") from None
    return (height, width)


def _not_a_readable_picture(path: str | Path) -> ImageError:
    """The refusal both PIL readers raise, worded once.

    `Image.open` identifies a file by its content, so this fires on a file whose *bytes* are
    not one of `READABLE_FORMATS` however it is named - a TIFF called `.png` reads fine and a
    PostScript file called `.png` does not. The message says so, because the first thing
    anybody does with "could not read" is look at the extension, and here the extension was
    never what decided.
    """
    return ImageError(
        f"could not read '{path}': its contents are not one of {', '.join(READABLE_FORMATS)}. "
        "The name is not what decides - PIL identifies a picture by its bytes - so a file "
        "carrying another format under a picture's extension is refused here rather than "
        "handed to whichever decoder its contents ask for"
    )


def resize_image(array: np.ndarray, size: tuple[int, ...], *, nearest: bool = False) -> np.ndarray:
    """Resize a channels-last 2D float array. The rank-2 counterpart of `resize_volume`."""
    return _resize_array(array, tuple(size), nearest=nearest)


def _resize_array(array: np.ndarray, size: tuple[int, ...], *, nearest: bool) -> np.ndarray:
    """Resize a float array through PIL, one channel at a time.

    DICOM arrives as real values rather than bytes, so it cannot go through the 8-bit path
    without losing exactly the precision the modality LUT was applied to recover.
    """
    if len(size) != 2:
        raise ImageError(f"2D readers take a (height, width), got {tuple(size)}")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    channels = [
        np.asarray(
            Image.fromarray(array[..., c].astype(np.float32), mode="F").resize(
                (size[1], size[0]), resample=resample
            ),
            dtype=np.float32,
        )
        for c in range(array.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def to_float(array: np.ndarray) -> np.ndarray:
    """0-255 integers to 0-1 floats, leaving anything already floating alone."""
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    return array.astype(np.float32) / 255.0


def tile(array: np.ndarray, splits: tuple[int, ...]) -> np.ndarray:
    """Cut a channels-last array into a grid of tiles.

    Returns shape (n_tiles, *tile_shape, channels), in row-major order: for splits (2, 3)
    the tiles come out as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2). `stitch` relies on
    that order.
    """
    rank = len(splits)
    spatial = array.shape[:rank]
    if array.ndim != rank + 1:
        raise ImageError(
            f"expected {rank} spatial dimensions plus channels, got shape {array.shape}"
        )
    bad = [(size, n) for size, n in zip(spatial, splits, strict=True) if size % n]
    if bad:
        raise ImageError(f"cannot cut {spatial} into {splits}: every dimension must divide exactly")

    tile_shape = tuple(size // n for size, n in zip(spatial, splits, strict=True))
    channels = array.shape[-1]

    # (s0, t0, s1, t1, ..., C) -> (s0, s1, ..., t0, t1, ..., C)
    interleaved = []
    for n, t in zip(splits, tile_shape, strict=True):
        interleaved.extend((n, t))
    reshaped = array.reshape(*interleaved, channels)
    order = [2 * i for i in range(rank)] + [2 * i + 1 for i in range(rank)] + [2 * rank]
    return reshaped.transpose(order).reshape(math.prod(splits), *tile_shape, channels)


def stitch(tiles: np.ndarray, splits: tuple[int, ...]) -> np.ndarray:
    """Put tiles produced by `tile` back into one array. The inverse of `tile`."""
    rank = len(splits)
    expected = math.prod(splits)
    if tiles.shape[0] != expected:
        raise ImageError(f"splits {splits} needs {expected} tiles, got {tiles.shape[0]}")
    if tiles.ndim != rank + 2:
        raise ImageError(
            f"expected tiles shaped (n, {'x'.join('t' * rank)}, channels), got {tiles.shape}"
        )

    tile_shape = tiles.shape[1:-1]
    channels = tiles.shape[-1]
    grouped = tiles.reshape(*splits, *tile_shape, channels)
    # (s0, s1, ..., t0, t1, ..., C) -> (s0, t0, s1, t1, ..., C)
    order: list[int] = []
    for i in range(rank):
        order.extend((i, rank + i))
    order.append(2 * rank)
    full = tuple(n * t for n, t in zip(splits, tile_shape, strict=True))
    return grouped.transpose(order).reshape(*full, channels)
