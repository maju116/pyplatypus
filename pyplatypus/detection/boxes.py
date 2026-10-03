"""Getting an image to the network's input size without lying about its shape.

The package this replaces resized straight to `(net_h, net_w)`, so a 640x480 photograph
arrived stretched. That is internally consistent - boxes were normalised by the source
dimensions, so the targets matched the stretched image and training worked - but it costs
two things. The COCO weights were trained on letterboxed input, so loading them onto
stretched images is a silent mismatch. And a circular cell in a non-square image becomes
an ellipse, which is information thrown away for nothing.

**Letterboxing is the default here**: scale by one factor, centre, pad the rest. The pad
colour is grey rather than black because black is a legitimate pixel value in a
radiograph, and a model that learns "the dark band at the edge means nothing" has learned
something about the padding rather than the anatomy.

Every transform is reversible, and `Letterbox` carries what it did so a box predicted in
the network's space can be put back on the pixels it came from. That is the detection
equivalent of `predict(space="source")`, and without it a box cannot be drawn on the
original image at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyplatypus.detection.metrics import DetectionError


@dataclass(frozen=True)
class Letterbox:
    """What was done to an image to make it the network's size.

    `scale` is the single factor both axes were multiplied by; `pad_x` and `pad_y` are the
    pixels added on the left and top. Stored rather than recomputed because the inverse
    has to undo exactly what the forward pass did - deriving it again from the shapes is
    how a rounding difference becomes a box half a pixel out.
    """

    scale: float
    pad_x: float
    pad_y: float
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]

    @classmethod
    def fit(cls, source_shape: tuple[int, int],
            target_shape: tuple[int, int]) -> Letterbox:
        """The transform that fits `source_shape` inside `target_shape`, centred."""
        source_h, source_w = (int(source_shape[0]), int(source_shape[1]))
        target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
        if source_h <= 0 or source_w <= 0:
            raise DetectionError(
                f"an image cannot have shape {(source_h, source_w)}"
            )
        if target_h <= 0 or target_w <= 0:
            raise DetectionError(
                f"a network input cannot have shape {(target_h, target_w)}"
            )
        scale = min(target_h / source_h, target_w / source_w)
        return cls(
            scale=scale,
            pad_x=(target_w - source_w * scale) / 2,
            pad_y=(target_h - source_h * scale) / 2,
            source_shape=(source_h, source_w),
            target_shape=(target_h, target_w),
        )

    def forward(self, boxes) -> np.ndarray:
        """Source-pixel boxes to network-input boxes."""
        array = _boxes(boxes)
        if array.size == 0:
            return array
        out = array * self.scale
        out[:, [0, 2]] += self.pad_x
        out[:, [1, 3]] += self.pad_y
        return out

    def inverse(self, boxes) -> np.ndarray:
        """Network-input boxes back to source pixels.

        Clipped to the source image, because a box may be predicted partly inside the
        padding - which is not a place, so a coordinate there means nothing.
        """
        array = _boxes(boxes)
        if array.size == 0:
            return array
        out = array.copy()
        out[:, [0, 2]] -= self.pad_x
        out[:, [1, 3]] -= self.pad_y
        out /= self.scale
        height, width = self.source_shape
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
        return out

    def apply_to_image(self, image, fill: float = 0.5) -> np.ndarray:
        """The image itself, scaled and padded. Channels last, any number of them.

        Nearest-neighbour is deliberate for the resize of a *mask-like* array and wrong for
        a photograph, so this does bilinear through the same reader the rest of the package
        uses rather than inventing a second resize.
        """
        from pyplatypus.data.images import resize_image

        array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise DetectionError(
                f"an image must be 2 or 3 dimensional, channels last; got shape "
                f"{array.shape}"
            )
        source_h, source_w = self.source_shape
        if array.shape[:2] != (source_h, source_w):
            raise DetectionError(
                f"this Letterbox was fitted for {(source_h, source_w)} but the image is "
                f"{array.shape[:2]}; fit a new one rather than reusing this"
            )

        inner_h = max(1, round(source_h * self.scale))
        inner_w = max(1, round(source_w * self.scale))
        resized = resize_image(array, (inner_h, inner_w))

        target_h, target_w = self.target_shape
        out = np.full((target_h, target_w, array.shape[2]), fill, dtype=np.float32)
        top = round(self.pad_y)
        left = round(self.pad_x)
        out[top:top + inner_h, left:left + inner_w] = resized
        return out


def _boxes(boxes) -> np.ndarray:
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 4), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 4:
        raise DetectionError(
            f"boxes must be an (n, 4) array of (xmin, ymin, xmax, ymax); got shape "
            f"{np.asarray(boxes).shape}"
        )
    return array


def clip_boxes(boxes, shape: tuple[int, int]) -> np.ndarray:
    """Boxes trimmed to an image's bounds."""
    array = _boxes(boxes).copy()
    if array.size == 0:
        return array
    height, width = int(shape[0]), int(shape[1])
    array[:, [0, 2]] = np.clip(array[:, [0, 2]], 0, width)
    array[:, [1, 3]] = np.clip(array[:, [1, 3]], 0, height)
    return array


def box_areas(boxes) -> np.ndarray:
    array = _boxes(boxes)
    if array.size == 0:
        return np.zeros(0, dtype=float)
    return np.clip(array[:, 2] - array[:, 0], 0, None) * \
        np.clip(array[:, 3] - array[:, 1], 0, None)


def drop_degenerate(boxes, labels, *, minimum_side: float = 1.0):
    """Boxes too small to be a box, and their labels, removed.

    A letterbox that shrinks an image by 8 can take a 4-pixel platelet below one pixel, at
    which point it is not an object any more and training on it teaches the model that
    nothing is something. Returned rather than refused, with the count, because a dataset
    containing a few is normal and stopping on the first turns an afternoon into a week.
    """
    array = _boxes(boxes)
    tags = np.asarray(labels).ravel()
    if len(array) != len(tags):
        raise DetectionError(
            f"{len(array)} boxes and {len(tags)} labels; they must agree"
        )
    if array.size == 0:
        return array, tags, 0
    keep = ((array[:, 2] - array[:, 0]) >= minimum_side) & \
           ((array[:, 3] - array[:, 1]) >= minimum_side)
    return array[keep], tags[keep], int((~keep).sum())
