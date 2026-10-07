"""Reading boxes off disk, and the one-pixel question nobody asks.

**Pascal VOC stores 1-based inclusive pixel indices.** A box covering pixels 1 to 10 is
ten pixels wide, and `xmax - xmin` is nine. Read naively as continuous coordinates it
loses a pixel on each side, which for a 10-pixel object leaves **81% of its true area** -
and on BCCD a platelet is about fifteen pixels across, so this is not a rounding question.
VOC's own evaluator compensates by adding 1 back inside the IoU; this package keeps its
metrics in continuous coordinates instead and fixes the boxes once, here, where the format
is known.

The conversion is to subtract 1 from the minima only: pixels 1..10 become the continuous
span [0, 10], whose width is 10.

`coordinates="voc"` does that and is the default, because it is what the format specifies.
`coordinates="zero_based"` leaves the numbers alone, for the many tools that write VOC's
XML shape with 0-based coordinates. **Choosing wrongly is caught rather than silent**: a
minimum of 0 cannot occur under a 1-based reading, so a file containing one is refused
with the alternative named. The reverse - 1-based data read as 0-based - cannot be proven
from one file, so `describe_annotations` reports the evidence and leaves the judgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np

from pyplatypus.detection.metrics import DetectionError

Coordinates = Literal["voc", "zero_based"]


@dataclass(frozen=True)
class Annotation:
    """One image's boxes, in continuous source-pixel coordinates."""

    path: Path
    width: int
    height: int
    boxes: np.ndarray  # (n, 4) xmin, ymin, xmax, ymax
    labels: np.ndarray  # (n,) int, indices into the label list
    names: list[str] = field(default_factory=list)
    difficult: np.ndarray | None = None
    image_path: str | None = None

    def as_truth(self) -> dict[str, Any]:
        """The shape `detection_report` wants."""
        out: dict[str, Any] = {"boxes": self.boxes, "labels": self.labels}
        if self.difficult is not None:
            out["difficult"] = self.difficult
        return out


def read_voc(
    path, labels: list[str], *, coordinates: Coordinates = "voc", strict_labels: bool = True
) -> Annotation:
    """One Pascal VOC XML file.

    `strict_labels=False` skips objects whose class is not in `labels`, which is how you
    train on two of a dataset's three classes. True by default, because a label the list
    does not contain is far more often a typo in the list than an intention.
    """
    file = Path(path)
    try:
        root = ElementTree.parse(file).getroot()
    except ElementTree.ParseError as error:
        raise DetectionError(f"'{file}' is not readable XML: {error}") from error

    size = root.find("size")
    if size is None:
        raise DetectionError(f"'{file}' has no <size>, so its boxes have no frame")
    width = _as_int(size.findtext("width"), file, "width")
    height = _as_int(size.findtext("height"), file, "height")

    boxes, indices, names, difficult = [], [], [], []
    for node in root.findall("object"):
        name = (node.findtext("name") or "").strip()
        if name not in labels:
            if strict_labels:
                raise DetectionError(
                    f"'{file}' contains the class '{name}', which is not in the label "
                    f"list {labels}. Add it, or pass strict_labels=False to train on a "
                    f"subset of the classes."
                )
            continue
        box_node = node.find("bndbox")
        if box_node is None:
            raise DetectionError(f"'{file}' has an <object> with no <bndbox>")
        raw = [
            _as_float(box_node.findtext(corner), file, corner)
            for corner in ("xmin", "ymin", "xmax", "ymax")
        ]
        boxes.append(_to_continuous(raw, coordinates, file))
        indices.append(labels.index(name))
        names.append(name)
        difficult.append(_as_int(node.findtext("difficult") or "0", file, "difficult") != 0)

    return Annotation(
        path=file,
        width=width,
        height=height,
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        labels=np.asarray(indices, dtype=int),
        names=names,
        difficult=np.asarray(difficult, dtype=bool),
        image_path=root.findtext("filename"),
    )


def read_labelme(path, labels: list[str], *, strict_labels: bool = True) -> Annotation:
    """One LabelMe JSON file.

    LabelMe draws polygons as well as rectangles. **A polygon becomes its bounding box**,
    which is the only reading detection can use and is stated here because it is a loss:
    the shape is discarded and only its extent survives.
    """
    file = Path(path)
    try:
        payload = json.loads(file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DetectionError(f"'{file}' is not readable JSON: {error}") from error

    height = payload.get("imageHeight")
    width = payload.get("imageWidth")
    if not height or not width:
        raise DetectionError(f"'{file}' has no imageHeight/imageWidth, so its boxes have no frame")

    boxes, indices, names = [], [], []
    for shape in payload.get("shapes", []):
        name = str(shape.get("label", "")).strip()
        if name not in labels:
            if strict_labels:
                raise DetectionError(
                    f"'{file}' contains the class '{name}', which is not in the label "
                    f"list {labels}. Add it, or pass strict_labels=False."
                )
            continue
        points = np.asarray(shape.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            raise DetectionError(
                f"'{file}' has a shape labelled '{name}' with points of shape "
                f"{points.shape}; a box needs at least two (x, y) pairs"
            )
        # LabelMe's coordinates are already continuous, and its two rectangle points are
        # opposite corners in no guaranteed order.
        boxes.append(
            [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]
        )
        indices.append(labels.index(name))
        names.append(name)

    return Annotation(
        path=file,
        width=int(width),
        height=int(height),
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        labels=np.asarray(indices, dtype=int),
        names=names,
        difficult=np.zeros(len(boxes), dtype=bool),
        image_path=payload.get("imagePath"),
    )


def read_annotations(
    directory,
    labels: list[str],
    *,
    annotation_format: Literal["pascal_voc", "labelme"] = "pascal_voc",
    coordinates: Coordinates = "voc",
    strict_labels: bool = True,
) -> list[Annotation]:
    """Every annotation in a directory, in sorted order so a split is reproducible."""
    root = Path(directory)
    if not root.is_dir():
        raise DetectionError(f"'{root}' is not a directory")
    pattern = "*.xml" if annotation_format == "pascal_voc" else "*.json"
    files = sorted(root.glob(pattern))
    if not files:
        raise DetectionError(
            f"'{root}' holds no {pattern} files. The format is '{annotation_format}'; "
            f"pass annotation_format='labelme' for JSON."
        )
    if annotation_format == "pascal_voc":
        return [
            read_voc(f, labels, coordinates=coordinates, strict_labels=strict_labels) for f in files
        ]
    return [read_labelme(f, labels, strict_labels=strict_labels) for f in files]


def describe_annotations(
    directory,
    labels: list[str],
    *,
    annotation_format: Literal["pascal_voc", "labelme"] = "pascal_voc",
    limit: int = 200,
) -> dict[str, Any]:
    """What is in a directory of annotations, and which coordinate convention fits.

    Reads the numbers exactly as written - no conversion - so the evidence is the data's
    own. `minimum_seen == 0` proves the file is 0-based, because a 1-based index cannot be
    zero. Everything else is suggestive at best, which is why this reports rather than
    decides.
    """
    root = Path(directory)
    pattern = "*.xml" if annotation_format == "pascal_voc" else "*.json"
    files = sorted(root.glob(pattern))[:limit]
    if not files:
        raise DetectionError(f"'{root}' holds no {pattern} files")

    reader = (
        (lambda f: read_voc(f, labels, coordinates="zero_based", strict_labels=False))
        if annotation_format == "pascal_voc"
        else (lambda f: read_labelme(f, labels, strict_labels=False))
    )

    per_class: dict[str, int] = {name: 0 for name in labels}
    minima, widths, heights, sides = [], [], [], []
    touching_edge = 0
    for file in files:
        annotation = reader(file)
        for name in annotation.names:
            per_class[name] = per_class.get(name, 0) + 1
        if annotation.boxes.size:
            minima.append(float(min(annotation.boxes[:, 0].min(), annotation.boxes[:, 1].min())))
            sides.extend((annotation.boxes[:, 2] - annotation.boxes[:, 0]).tolist())
            sides.extend((annotation.boxes[:, 3] - annotation.boxes[:, 1]).tolist())
            touching_edge += int(
                (
                    (annotation.boxes[:, 2] >= annotation.width)
                    | (annotation.boxes[:, 3] >= annotation.height)
                ).sum()
            )
        widths.append(annotation.width)
        heights.append(annotation.height)

    minimum_seen = min(minima) if minima else None
    return {
        "files": len(files),
        "objects": int(sum(per_class.values())),
        "per_class": per_class,
        "image_widths": sorted(set(widths)),
        "image_heights": sorted(set(heights)),
        "minimum_coordinate": minimum_seen,
        "boxes_touching_the_far_edge": touching_edge,
        "smallest_side": float(min(sides)) if sides else None,
        "median_side": float(np.median(sides)) if sides else None,
        # The only part of this that is proof rather than evidence.
        "coordinates": (
            "zero_based"
            if minimum_seen == 0
            else "consistent with voc (1-based); no zero minimum seen"
        ),
    }


def _to_continuous(raw: list[float], coordinates: Coordinates, file: Path) -> list[float]:
    if coordinates == "zero_based":
        return raw
    if coordinates != "voc":
        raise DetectionError(f"coordinates must be 'voc' or 'zero_based'; got {coordinates!r}")
    xmin, ymin, xmax, ymax = raw
    if xmin == 0 or ymin == 0:
        raise DetectionError(
            f"'{file}' has a box with a minimum of 0, which a 1-based index cannot be, so "
            f"this file is 0-based and coordinates='voc' would make it -1. Pass "
            f"coordinates='zero_based'. `describe_annotations` reports what a directory "
            f"looks like."
        )
    # Minima only: pixels 1..10 inclusive become the continuous span [0, 10], width 10.
    return [xmin - 1, ymin - 1, xmax, ymax]


def _as_int(text: str | None, file: Path, what: str) -> int:
    return round(_as_float(text, file, what))


def _as_float(text: str | None, file: Path, what: str) -> float:
    if text is None or not str(text).strip():
        raise DetectionError(f"'{file}' has no <{what}>")
    try:
        return float(str(text).strip())
    except ValueError as error:
        raise DetectionError(
            f"'{file}' has <{what}>{text}</{what}>, which is not a number"
        ) from error
