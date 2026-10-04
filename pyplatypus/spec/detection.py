"""Boxes rather than masks: the detection half of the specification.

What is striking about writing this is how little is new. `DataSpec` already said where
the data is and `ModelSpec` already said how long to train for; detection adds two
questions neither task can answer for the other - what the classes are called, and what
shape the boxes the model guesses from are.

Three fields here exist because the BCCD run needed them and nothing in the spec could
say them: the anchors, the coordinate convention of the annotation files, and the two
thresholds that turn a grid of logits into a list of boxes. They were command-line flags
in `examples/detect_blood_cells.py`, which is where a setting lives when it has nowhere
better to be.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from pyplatypus.spec.data import DataSpec
from pyplatypus.spec.models import ModelSpec

AnchorGroup = tuple[tuple[float, float], ...]


def _strides() -> tuple[int, ...]:
    """YOLOv3's strides, from the one place they are written.

    Imported on use rather than at module level, because `encode` imports numpy and no
    module in the spec layer does: validating a configuration is what happens before
    anything is read, and it should not need the data stack to be importable.
    `tests/spec/test_spec_layer_is_light.py` holds that line.

    Both the number of grids and the coarsest stride come from here. Two constants saying
    "three grids" is one of them going stale.
    """
    from pyplatypus.detection.encode import STRIDES

    return STRIDES


class DetectionArchitecture(str, Enum):
    """One member, and an enum anyway.

    A `Literal["yolo3"]` would say the same thing today and would have to become an enum
    the moment a second detector arrives, changing the type of a field the R package
    sends. See `Architecture` for the same reasoning on the segmentation side.
    """

    YOLO3 = "yolo3"


class DetectionData(DataSpec):
    """Where the images and their annotation files are.

    `classes` is detection's answer to the colormap, and it is stated for the same reason:
    the position of a name in this list *is* the class index the model learns. Read from
    the annotation files instead, the order would come out alphabetical - reproducible,
    and unrelated to anything. A model trained with `RBC` at index 0 and used where index
    0 means `WBC` reports plausible boxes with plausible scores and nothing downstream can
    tell.

    Unlike a segmentation colormap there is no background entry. A pixel always belongs to
    some class, so segmentation needs a name for "none of them"; a region of an image that
    holds no object is simply not a box.
    """

    subdirs: tuple[str, str] = ("images", "annotations")

    classes: list[str] = Field(
        min_length=1,
        description=(
            "Class names in class order. Position is the class index. No background "
            "entry - an image region with no object is not a box."
        ),
    )
    annotation_format: Literal["pascal_voc", "labelme"] = Field(
        "pascal_voc",
        description=(
            "Pascal VOC XML, the format BCCD and most detection datasets ship, or LabelMe "
            "JSON. Both readers are in `pyplatypus.detection.annotations`."
        ),
    )
    coordinates: Literal["voc", "zero_based"] | None = Field(
        None,
        description=(
            "How to read a Pascal VOC box. The format's own convention is 1-based and "
            "inclusive of both endpoints, so a 10-pixel-wide box runs xmin=1 xmax=10 and "
            "the width is `xmax - xmin + 1`. Plenty of tools write the same files 0-based "
            "and exclusive, where that box is xmin=0 xmax=10. Reading one as the other "
            "shifts every box by a pixel and shrinks it by one, which is invisible at a "
            "glance and costs real IoU on small objects.\n"
            "Unset means 'voc'. `describe_annotations` reports the evidence: a minimum "
            "coordinate of 0 proves the file is 0-based, since a 1-based index cannot be "
            "zero."
        ),
    )
    strict_labels: bool = Field(
        True,
        description=(
            "Refuse an object whose class is not in `classes`. False skips it instead, "
            "which is how you train on two of a dataset's three classes. True by default: "
            "an unexpected label is more often a typo in `classes` than an intention."
        ),
    )

    @model_validator(mode="after")
    def coordinates_are_a_voc_question(self):
        """LabelMe stores continuous pixel coordinates, so there is no convention to pick.
        Accepting the field and ignoring it would leave someone believing they had changed
        how their boxes are read."""
        if self.coordinates is not None and self.annotation_format != "pascal_voc":
            raise ValueError(
                f"`coordinates` describes how to read a Pascal VOC box, but "
                f"annotation_format is '{self.annotation_format}', which stores "
                f"continuous pixel coordinates and needs no convention"
            )
        return self

    @model_validator(mode="after")
    def class_names_are_distinct(self):
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(
                "class names must be distinct - two classes sharing a name cannot be "
                "told apart, and one of them would never be matched"
            )
        return self

    @property
    def label_column(self) -> str:
        return "annotations"

    @property
    def n_class(self) -> int:
        """How many classes there are. `classes` decides; nothing else."""
        return len(self.classes)

    @property
    def voc_coordinates(self) -> str:
        """The convention to read with, with the default resolved."""
        return self.coordinates or "voc"


class DetectionModel(ModelSpec):
    """One detector.

    No `n_class`: the data's `classes` is the only place it is written. Segmentation
    carries it on the model too and cross-checks the two, which is a consequence of its
    history rather than a design - one source cannot disagree with itself.
    """

    architecture: DetectionArchitecture = DetectionArchitecture.YOLO3

    anchors: list[AnchorGroup] | None = Field(
        None,
        description=(
            "The box shapes the model predicts offsets from, as fractions of the input "
            "size, one group per grid from coarsest to finest. Unset fits them to the "
            "training annotations with k-means under an IoU distance, which is what you "
            "want: COCO's anchors on blood cells cover their boxes at a mean IoU of 0.67 "
            "against 0.92 for anchors fitted to them. Fitted anchors are recorded with "
            "the run, since a detector cannot be reloaded without them."
        ),
    )
    anchors_per_grid: int = Field(
        3,
        ge=1,
        description=(
            "How many anchors to fit per grid, when `anchors` is not given. It also sets "
            "the head's width, which is anchors_per_grid * (n_class + 5)."
        ),
    )

    ignore_threshold: float = Field(
        0.5,
        gt=0,
        le=1,
        description=(
            "A cell that is not responsible for an object but whose box overlaps one this "
            "much is left unsupervised rather than trained towards 'no object'. Without "
            "it a nearly-correct guess beside the responsible cell is punished, which is "
            "the one thing a detector should not learn."
        ),
    )

    score_threshold: float = Field(
        0.01,
        ge=0,
        lt=1,
        description=(
            "Discard a predicted box below this confidence. Low on purpose: average "
            "precision is computed over the whole ranking, so a cut here is a ceiling on "
            "recall that no threshold chosen later can lift."
        ),
    )
    nms_threshold: float = Field(
        0.45,
        gt=0,
        le=1,
        description=(
            "Two boxes of the same class overlapping this much are one object, and the "
            "lower-scoring one is dropped. Per class, so a platelet sitting on a red "
            "cell survives."
        ),
    )
    operating_point: float = Field(
        0.5,
        gt=0,
        lt=1,
        description=(
            "The confidence at which precision and recall are reported. Separate from "
            "`score_threshold` because they answer different questions: that one decides "
            "what is computed at all, this one decides where on the curve to stand."
        ),
    )

    def weights_fingerprint(self) -> dict:
        """`anchors_per_grid` rather than `blocks` and `filters`: it is what sets the
        head's width, so weights with a different one cannot be loaded at all. `n_class`
        sets it too and is not here, because the model does not hold it - the engine adds
        it, from the data's `classes`."""
        return {**super().weights_fingerprint(),
                "anchors_per_grid": self.anchors_per_grid}

    min_visibility: float = Field(
        0.25,
        ge=0,
        le=1,
        description=(
            "How much of a box must survive an augmentation that removes part of the "
            "frame, as a fraction of its original area, for the box to be kept. A "
            "convention rather than a measurement; what is defensible is the direction. "
            "A box keeping two pixels of a cell teaches the model that a two-pixel "
            "fragment is a whole cell, which produces false positives everywhere, while "
            "dropping a heavily truncated object only fails to teach it about that "
            "object. Pascal VOC marks such objects `truncated` for the same reason.\n"
            "Irrelevant unless a transform can lose part of the frame: flips and "
            "rotations never do."
        ),
    )

    @model_validator(mode="after")
    def not_fitting_needs_weights(self):
        """Loading a detector is not loading a segmentation model, and the difference is
        the anchors.

        Named after the base's validator on purpose, which replaces it: `fit: false` would
        otherwise be told "fit=false only makes sense together with weights", and adding
        weights to satisfy that would arrive here. One dead end is better reached in one
        step than in two.

        Weights decode boxes *relative to the anchors they were trained with*. Read with
        other anchors, the same weights produce boxes scaled by a fixed factor - plausible
        boxes, plausible scores, wrong places, and nothing in any output says so. So
        loading has to adopt the anchors recorded beside the file, and deciding what
        happens when the specification also names some is a decision that has not been
        taken. Until it is, this refuses rather than guessing.

        `export_weights` works: writing records the anchors in the sidecar, which is what
        makes the decision possible later.
        """
        if self.weights is None and self.fit:
            return self
        asked = "weights" if self.weights is not None else "fit=false"
        raise ValueError(
            f"{asked}: loading weights into a detector is not supported yet. A "
            f"detector's weights only mean anything together with the anchors they were "
            f"trained with - read with other anchors they give plausible boxes in the "
            f"wrong places, silently - so loading has to adopt the anchors from the "
            f"file, and that is not built. Training records them: "
            f"DetectionEngine.export_weights writes the anchors into the sidecar."
        )

    @model_validator(mode="after")
    def detection_is_two_dimensional(self):
        """Refused while the spec is read rather than at build time, so a 3D detection
        spec fails before anything is allocated. YOLOv3 is a 2D architecture; boxes in a
        volume are a different model, not this one with an extra axis."""
        if self.rank != 2:
            raise ValueError(
                f"detection is 2D: input_shape must be (height, width), got "
                f"{tuple(self.input_shape)}. Boxes in a volume need a 3D detector, which "
                f"this is not."
            )
        return self

    @model_validator(mode="after")
    def input_divides_by_the_coarsest_stride(self):
        """The three grids are the input divided by 32, 16 and 8. A size that is not a
        multiple of 32 gives a coarsest grid that does not tile the image, so the boxes
        decoded from it sit slightly off everywhere."""
        coarsest = max(_strides())
        bad = [size for size in self.input_shape if size % coarsest]
        if bad:
            raise ValueError(
                f"every side of input_shape must be divisible by {coarsest}; "
                f"{tuple(self.input_shape)} is not"
            )
        return self

    @model_validator(mode="after")
    def anchors_describe_the_grids(self):
        if self.anchors is None:
            return self
        scales = len(_strides())
        if len(self.anchors) != scales:
            raise ValueError(
                f"anchors needs {scales} groups, one per grid, coarsest first; got "
                f"{len(self.anchors)}"
            )
        widths = {len(group) for group in self.anchors}
        if len(widths) > 1:
            listed = ", ".join(str(len(group)) for group in self.anchors)
            raise ValueError(
                f"every grid needs the same number of anchors - the head is one tensor "
                f"of width anchors_per_grid * (n_class + 5); got {listed}"
            )
        given = widths.pop()
        if given != self.anchors_per_grid:
            raise ValueError(
                f"anchors gives {given} per grid but anchors_per_grid is "
                f"{self.anchors_per_grid}; leave anchors_per_grid unset, or make them agree"
            )
        for group in self.anchors:
            for pair in group:
                if not all(0 < value <= 1 for value in pair):
                    raise ValueError(
                        f"anchors are fractions of the input size, so each must be above "
                        f"0 and at most 1; got {pair}. An anchor in pixels - (116, 90) "
                        f"for COCO at 416 - is this times the input size."
                    )
        return self

    @property
    def anchors_as_tuples(self) -> tuple[AnchorGroup, ...] | None:
        """The anchors in the shape `encode` and `Yolo3Loss` take."""
        if self.anchors is None:
            return None
        return tuple(tuple((float(a), float(b)) for a, b in group) for group in self.anchors)
