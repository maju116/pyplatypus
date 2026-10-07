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
    def fit(cls, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Letterbox:
        """The transform that fits `source_shape` inside `target_shape`, centred."""
        source_h, source_w = (int(source_shape[0]), int(source_shape[1]))
        target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
        if source_h <= 0 or source_w <= 0:
            raise DetectionError(f"an image cannot have shape {(source_h, source_w)}")
        if target_h <= 0 or target_w <= 0:
            raise DetectionError(f"a network input cannot have shape {(target_h, target_w)}")
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
                f"an image must be 2 or 3 dimensional, channels last; got shape {array.shape}"
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
        out[top : top + inner_h, left : left + inner_w] = resized
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


def crop_boxes(
    image,
    boxes,
    *,
    context: float = 0.0,
    size: tuple[int, int] | None = None,
    fit: str = "letterbox",
    fill: float = 0.5,
) -> list[np.ndarray]:
    """One sub-image per box: detect, cut out, hand to something else.

    The pipeline this exists for is detection followed by a classifier the detector does
    not have - a model that finds objects in 3 classes, cropped and passed to one that
    knows 80. `predict` returns boxes in each image's own pixels precisely so that they
    can be used on the photograph they came from, and this is what uses them.

    **Always a list, even when `size` makes every crop the same shape.** A return type
    that depends on an argument makes every caller branch; `np.stack(crops)` is the batch
    and is one line. Boxes also differ in number per image, so a stack is not the general
    case anyway.

    Four decisions, each of which taken the other way quietly changes what a downstream
    model sees:

    - **Rounded outward** - the left and top floored, the right and bottom ceiled - so the
      crop contains the whole box. Rounding to nearest loses up to a pixel on each side,
      which on a 26-pixel platelet is a twelfth of it.
    - **`context` expands the box by a fraction of its own size** before cropping, and
      defaults to 0. A detector's box is tight by construction, and a classifier trained
      on photographs of whole objects does worse on something cut exactly at the edge -
      but expanding by default would mean every crop containing things nobody asked for.
      Scaled by the box, not in pixels, so one value suits a platelet and a white cell.
    - **`fit="letterbox"` when `size` is given**, not a stretch. Resizing a tall box to a
      square changes the aspect of every non-square object, and a classifier then sees a
      shape that does not occur in nature. `fit="stretch"` is there for models trained
      that way, which is many of them, and has to be asked for.
    - **A box with no overlap at all is an error**, named by index, because an empty array
      is not a thing to hand a classifier. Note that this is *not* a claim that such a box
      is impossible: a model is free to predict one off the frame and `predict` does not
      clip, so at a low confidence threshold some come back with no overlap once the
      letterbox is undone. `DetectionEngine.crops` therefore clips and `drop_degenerate`s
      first - the same two steps `DetectionDataset.read` already applies to the truths -
      and that is the fix for anyone meeting this with predictions of their own.

    Args:
        image: `height x width x channels` or `height x width`, channels last.
        boxes: `(n, 4)` corner boxes `(x1, y1, x2, y2)` in this image's pixels.
        context: expand each box by this fraction of its width and height per side.
        size: `(height, width)` every crop is brought to, or None to keep each as cut.
        fit: `"letterbox"` to preserve aspect by padding, `"stretch"` to resize both axes.
        fill: the padding value for `"letterbox"`, in the image's own units.

    Returns:
        One array per box, in the order given.
    """
    from pyplatypus.data.images import resize_image

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise DetectionError(
            f"an image must be 2 or 3 dimensional, channels last; got shape {array.shape}"
        )
    if fit not in ("letterbox", "stretch"):
        raise DetectionError(f"fit is 'letterbox' or 'stretch'; got {fit!r}")
    if context < 0:
        raise DetectionError(f"context cannot be negative; got {context}")

    wanted = _boxes(boxes)
    height, width = array.shape[0], array.shape[1]

    out = []
    for index, (x1, y1, x2, y2) in enumerate(wanted):
        if context:
            grow_x = (x2 - x1) * context
            grow_y = (y2 - y1) * context
            x1, x2 = x1 - grow_x, x2 + grow_x
            y1, y2 = y1 - grow_y, y2 + grow_y

        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))

        if right <= left or bottom <= top:
            raise DetectionError(
                f"box {index} is {(float(x1), float(y1), float(x2), float(y2))}, which "
                f"leaves nothing inside an image of {(height, width)}. An empty crop is "
                f"not a thing to hand a classifier, so this is refused rather than "
                f"returned. A *predicted* box can legitimately be off the frame - a model "
                f"is free to say so - which is why `DetectionEngine.crops` clips and then "
                f"`drop_degenerate`s before calling this, and why doing the same is the "
                f"fix if these are predictions."
            )

        crop = array[top:bottom, left:right]
        if size is not None:
            target = (int(size[0]), int(size[1]))
            if fit == "letterbox":
                crop = Letterbox.fit(crop.shape[:2], target).apply_to_image(crop, fill=fill)
            else:
                crop = resize_image(crop, target)
        out.append(crop)
    return out


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
    return np.clip(array[:, 2] - array[:, 0], 0, None) * np.clip(array[:, 3] - array[:, 1], 0, None)


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
        raise DetectionError(f"{len(array)} boxes and {len(tags)} labels; they must agree")
    if array.size == 0:
        return array, tags, 0
    keep = ((array[:, 2] - array[:, 0]) >= minimum_side) & (
        (array[:, 3] - array[:, 1]) >= minimum_side
    )
    return array[keep], tags[keep], int((~keep).sum())


def non_max_suppression(
    boxes,
    scores,
    labels=None,
    *,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.0,
    per_class: bool = True,
    limit: int | None = None,
):
    """Keep the confident box and drop the ones that overlap it.

    A detector predicts from every cell of every grid, so one object arrives as a cluster
    of boxes. Without this, each cluster is one true positive and a handful of false ones,
    and precision collapses for a reason that has nothing to do with whether the object was
    found - which is the same thing the metric's greedy matching says: a duplicate box is a
    false positive.

    **Suppression is per class by default.** A platelet sitting on a red cell is two
    objects at nearly the same place, and suppressing across classes deletes one of them.
    Pass `per_class=False` only when the classes are genuinely exclusive.
    """
    array = _boxes(boxes)
    confidence = np.asarray(scores, dtype=float).ravel()
    if len(array) != len(confidence):
        raise DetectionError(
            f"{len(array)} boxes and {len(confidence)} scores; there must be one score per box"
        )
    tags = (
        np.zeros(len(array), dtype=int) if labels is None else np.asarray(labels, dtype=int).ravel()
    )
    if len(tags) != len(array):
        raise DetectionError(f"{len(array)} boxes and {len(tags)} labels; they must agree")
    if not 0 < iou_threshold <= 1:
        raise DetectionError(f"iou_threshold must be above 0 and at most 1; got {iou_threshold}")
    if array.size == 0:
        return np.zeros(0, dtype=int)

    alive = confidence >= score_threshold
    groups = (
        [np.where(alive & (tags == value))[0] for value in np.unique(tags)]
        if per_class
        else [np.where(alive)[0]]
    )

    kept: list[int] = []
    for group in groups:
        order = group[np.argsort(-confidence[group], kind="stable")]
        while order.size:
            best = order[0]
            kept.append(int(best))
            if order.size == 1:
                break
            overlaps = _pairwise_iou(array[best], array[order[1:]])
            order = order[1:][overlaps < iou_threshold]

    kept_array = np.asarray(kept, dtype=int)
    # Back into descending confidence across classes, so a caller taking the first N takes
    # the most confident N rather than the most confident of class 0.
    kept_array = kept_array[np.argsort(-confidence[kept_array], kind="stable")]
    return kept_array[:limit] if limit is not None else kept_array


def _pairwise_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    left = np.maximum(one[0], many[:, 0])
    top = np.maximum(one[1], many[:, 1])
    right = np.minimum(one[2], many[:, 2])
    bottom = np.minimum(one[3], many[:, 3])
    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_one = max(one[2] - one[0], 0) * max(one[3] - one[1], 0)
    area_many = np.clip(many[:, 2] - many[:, 0], 0, None) * np.clip(
        many[:, 3] - many[:, 1], 0, None
    )
    union = area_one + area_many - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
