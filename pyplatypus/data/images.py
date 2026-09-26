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
from PIL import Image

from pyplatypus.errors import PlatypusError

# PIL talks in (width, height); every shape in this package is (height, width). Mixing
# them is the classic silent bug, so the conversion happens in exactly one place.
_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


class ImageError(PlatypusError):
    kind = "image_error"


def read_image(path: str | Path, *, channels: int = 3,
               size: tuple[int, ...] | None = None,
               nearest: bool = False, dicom_window="auto") -> np.ndarray:
    """Read one image or volume as a channels-last array.

    Scaled to 0-1 for anything read in real units (DICOM, NIfTI) and left at 0-255 for
    ordinary pictures, which `to_float` divides later.

    `nearest` must be used for masks: interpolating a mask invents colours that match no
    class, which then silently become background.

    DICOM is detected by content rather than by extension, because exports out of an
    archive frequently carry no extension or one the archive invented.
    """
    from pyplatypus.data.dicom import looks_like_dicom, read_dicom
    from pyplatypus.data.volumes import looks_like_volume, read_volume

    # A volume asks for a different reader, not a different pipeline. Dispatching here keeps
    # every caller - the dataset, the plotting helpers, the R surface - reading "whatever is
    # at this path, the way training would", which is the only way a picture of a prediction
    # can be trusted to show what the model saw.
    if looks_like_volume(path):
        return read_volume(path, window=dicom_window, size=size, nearest=nearest,
                           channels=channels)

    if looks_like_dicom(path):
        array = read_dicom(path, window=dicom_window, channels=channels)
        if size is not None:
            array = _resize_array(array, size, nearest=nearest)
        return array

    mode = _PIL_MODE.get(channels)
    if mode is None:
        raise ImageError(f"channels must be 1, 3 or 4 for ordinary images, got {channels}")

    try:
        with Image.open(path) as handle:
            picture = handle.convert(mode)
            if size is not None:
                if len(size) != 2:
                    raise ImageError(
                        f"2D readers take a (height, width), got {tuple(size)}"
                    )
                resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
                picture = picture.resize((size[1], size[0]), resample=resample)
            array = np.asarray(picture, dtype=np.uint8)
    except OSError as error:
        raise ImageError(f"could not read '{path}': {error}") from None

    if array.ndim == 2:
        array = array[..., None]
    return array


def _resize_array(array: np.ndarray, size: tuple[int, ...], *, nearest: bool
                  ) -> np.ndarray:
    """Resize a float array through PIL, one channel at a time.

    DICOM arrives as real values rather than bytes, so it cannot go through the 8-bit path
    without losing exactly the precision the modality LUT was applied to recover.
    """
    if len(size) != 2:
        raise ImageError(f"2D readers take a (height, width), got {tuple(size)}")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    channels = [
        np.asarray(
            Image.fromarray(array[..., c].astype(np.float32), mode="F")
            .resize((size[1], size[0]), resample=resample),
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
        raise ImageError(
            f"cannot cut {spatial} into {splits}: every dimension must divide exactly"
        )

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
        raise ImageError(
            f"splits {splits} needs {expected} tiles, got {tiles.shape[0]}"
        )
    if tiles.ndim != rank + 2:
        raise ImageError(
            f"expected tiles shaped (n, {'x'.join('t' * rank)}, channels), "
            f"got {tiles.shape}"
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
