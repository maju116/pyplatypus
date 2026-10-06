"""One sample in, one detection training example out.

The segmentation counterpart of this file turns an image and its masks into an image and a
one-hot array. Here the second half is three target tensors, one per grid, and getting
there involves two steps a segmentation pipeline never takes:

* **the letterbox**, which scales by one factor and pads rather than stretching. A plain
  resize is internally consistent as long as the boxes are scaled the same way, but it
  distorts the objects themselves, and published YOLOv3 weights were trained on
  letterboxed input. The transform is kept, not just applied, because a prediction has to
  come back out of it onto the pixels the user gave us.
* **the encoder**, which places each box in the cell and anchor responsible for it. This
  can lose an object - two boxes of one shape whose centres fall in the same cell share a
  slot - so the loss is measured and reported rather than discovered.

Numpy only, like the segmentation dataset. The torch adapter is a dozen lines and lives in
`pyplatypus.training.torch_data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyplatypus.data.augmentation import BoxAugmenter
from pyplatypus.data.images import read_image, to_float
from pyplatypus.data.paths import Sample
from pyplatypus.detection.annotations import Annotation, read_labelme, read_voc
from pyplatypus.detection.boxes import Letterbox, drop_degenerate
from pyplatypus.detection.encode import encode
from pyplatypus.errors import PlatypusError
from pyplatypus.spec.detection import DetectionData, DetectionModel


class DetectionDataError(PlatypusError):
    kind = "detection_data_error"


@dataclass(frozen=True)
class Example:
    """What one sample becomes, before anything is turned into a tensor.

    The letterbox and the annotation come along because evaluation needs them: a predicted
    box has to go back through `fit.inverse` to be comparable with the truth, which lives
    in the photograph's own pixels and never in the network's.
    """

    image: np.ndarray                      # (h, w, c), 0-1 floats
    boxes: np.ndarray                      # (n, 4) in the letterboxed frame
    labels: np.ndarray                     # (n,) class indices
    fit: Letterbox
    annotation: Annotation
    dropped: int                           # boxes too small to survive the letterbox


@dataclass(frozen=True)
class TargetSurvey:
    """What the encoder could and could not represent, over a whole split.

    Reported before training rather than accumulated during it. The first version of this
    kept counters on the dataset object, which `DataLoader` increments **inside worker
    processes**, so the main process read zeros and printed silence. Up front is also where
    the number is useful: learning that the target cannot hold a tenth of the objects is a
    reason to change `input_shape` before spending half an hour, not after.
    """

    placed: int
    unplaced: int
    dropped: int
    images: int

    @property
    def total(self) -> int:
        return self.placed + self.unplaced

    @property
    def unplaced_fraction(self) -> float:
        return self.unplaced / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, float]:
        return {"placed": self.placed, "unplaced": self.unplaced,
                "dropped": self.dropped, "images": self.images,
                "unplaced_fraction": self.unplaced_fraction}


class DetectionDataset:
    """Samples on disk, presented as (image, three targets) tuples.

    One sample is one example: unlike segmentation there is no tiling, because cutting an
    image into a grid cuts boxes in half and a half box is not a smaller object.
    """

    def __init__(self, samples: tuple[Sample, ...], model: DetectionModel,
                 data: DetectionData, *, anchors, only_images: bool = False,
                 augmenter: BoxAugmenter | None = None):
        if not samples:
            raise DetectionDataError("a detection dataset needs at least one sample")
        self.samples = samples
        self.model = model
        self.data = data
        self.augmenter = augmenter
        self.anchors = tuple(tuple(tuple(float(v) for v in pair) for pair in group)
                             for group in anchors)
        self.only_images = only_images
        self.input_shape = (int(model.input_shape[0]), int(model.input_shape[1]))
        # Every sample, now, rather than when one is first read: counting paths costs no
        # I/O, and the alternative is fitting anchors over the whole split and then
        # failing on the first batch.
        for sample in samples:
            self._one_image(sample)

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ reading
    def annotation(self, index: int) -> Annotation:
        """The boxes of one sample, in the source image's own pixels."""
        sample = self.samples[index]
        if not sample.masks:
            raise DetectionDataError(
                f"sample '{sample.key}' has no annotation file. In nested_dirs mode that "
                f"is the '{self.data.subdirs[1]}' directory; in config_file mode it is "
                f"the '{self.data.label_column}' column."
            )
        if len(sample.masks) > 1:
            listed = ", ".join(p.name for p in sample.masks[:4])
            raise DetectionDataError(
                f"sample '{sample.key}' has {len(sample.masks)} annotation files "
                f"({listed}). One image's boxes are one file - several would be several "
                f"opinions about the same image, and nothing here can choose between "
                f"them. Segmentation joins several masks into one; boxes cannot be joined "
                f"that way, because a box is not a region of the frame."
            )
        return self._read_annotation(sample.masks[0])

    def _read_annotation(self, path: Path) -> Annotation:
        classes = list(self.data.classes)
        if self.data.annotation_format == "labelme":
            return read_labelme(path, classes, strict_labels=self.data.strict_labels)
        return read_voc(path, classes, coordinates=self.data.voc_coordinates,
                        strict_labels=self.data.strict_labels)

    def source_image(self, index: int) -> np.ndarray:
        """One sample's image at its native size, before any letterbox.

        `read()` letterboxes and keeps only the result, which is right for the network and
        wrong for anything that works in the photograph's own pixels - cropping a detection
        out of the 416x416 version would cut a downscaled object and then scale it back up,
        two resizes where none is needed.

        Separate from `read()` but through the same reader and the same options, so a crop
        and a prediction cannot disagree about channels or the DICOM window. Reading an
        image twice through two paths is how a mask once ended up beside the tissue it
        labelled.
        """
        sample = self.samples[index]
        return to_float(read_image(sample.images[0], channels=self.model.channels,
                                   dicom_window=self.data.window))

    def read(self, index: int) -> Example:
        """One sample, letterboxed, with everything evaluation will need.

        The image is read at its native size and letterboxed, rather than read at the
        model's size: the reader would resize, and a resize is the distortion the
        letterbox exists to avoid.
        """
        sample = self.samples[index]
        image = to_float(read_image(sample.images[0], channels=self.model.channels,
                                    dicom_window=self.data.window))

        if self.only_images:
            height, width = image.shape[0], image.shape[1]
            fit = Letterbox.fit((height, width), self.input_shape)
            return Example(image=fit.apply_to_image(image),
                           boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int),
                           fit=fit, annotation=_empty_annotation(sample, height, width),
                           dropped=0)

        annotation = self.annotation(index)
        self._check_frame(sample, annotation, image)
        fit = Letterbox.fit((annotation.height, annotation.width), self.input_shape)
        boxes, labels, dropped = drop_degenerate(fit.forward(annotation.boxes),
                                                 annotation.labels)
        return Example(image=fit.apply_to_image(image), boxes=boxes, labels=labels,
                       fit=fit, annotation=annotation, dropped=dropped)

    def _one_image(self, sample: Sample) -> None:
        """One image per sample, said rather than assumed.

        Reading `images[0]` and ignoring the rest is the silent kind of wrong: a user with
        two files per sample gets a run that trains, scores and reports, on half their
        data, with nothing anywhere to say which half. Segmentation can take several files
        per sample - `channels_from` names one pattern per channel - and detection has no
        equivalent, so there is nothing to interpret a second file as.
        """
        if len(sample.images) > 1:
            listed = ", ".join(p.name for p in sample.images[:4])
            raise DetectionDataError(
                f"sample '{sample.key}' has {len(sample.images)} image files ({listed}). "
                f"Detection reads one image per sample, and taking the first and ignoring "
                f"the rest would train on part of your data without saying so. "
                f"Segmentation's `channels_from` has no detection counterpart yet; one "
                f"image to one annotation file is the only arrangement."
            )

    def _check_frame(self, sample: Sample, annotation: Annotation,
                     image: np.ndarray) -> None:
        """The annotation's <size> has to be the image's size.

        A box is a position in a frame, so a frame of the wrong size puts every box
        somewhere else. It happens for real: a dataset is resized and the XML files are
        copied along unchanged, and the result trains without complaint because every
        number involved is plausible.
        """
        height, width = int(image.shape[0]), int(image.shape[1])
        if (annotation.height, annotation.width) != (height, width):
            raise DetectionDataError(
                f"sample '{sample.key}': the annotation says the image is "
                f"{annotation.width}x{annotation.height} and the file is {width}x{height}. "
                f"A box is a position in a frame, so the boxes would land somewhere else. "
                f"Either the images were resized after they were annotated, or the "
                f"annotation belongs to another image."
            )

    # ------------------------------------------------------------------ examples
    def __getitem__(self, index: int):
        """(image, (target_coarse, target_medium, target_fine)), all channels-last."""
        example = self.read(index)
        if self.only_images:
            return example.image, None

        image, boxes, labels = example.image, example.boxes, example.labels
        if self.augmenter is not None:
            # After the letterbox, not before. The boxes are in the model's frame by now
            # and the image is the model's size, so a transform with a fixed output size
            # behaves the same for every sample. The cost, which is real: a crop treats
            # the grey padding as if it were image, so on a dataset of mixed aspect
            # ratios a crop can return a tile that is mostly padding.
            image, boxes, labels = self.augmenter(image, boxes, labels)

        encoded = encode(boxes, labels, anchors=self.anchors,
                         input_shape=self.input_shape, n_class=self.data.n_class)
        return image, tuple(encoded.targets)

    # ------------------------------------------------------------------ surveying
    def survey(self) -> TargetSurvey:
        """How many boxes the target can hold, over every sample in this split.

        Reads the annotations and encodes them; it never reads a pixel, so it costs
        seconds on a dataset that takes minutes an epoch.
        """
        placed = unplaced = dropped = 0
        for index in range(len(self)):
            annotation = self.annotation(index)
            fit = Letterbox.fit((annotation.height, annotation.width), self.input_shape)
            boxes, labels, lost = drop_degenerate(fit.forward(annotation.boxes),
                                                  annotation.labels)
            dropped += lost
            encoded = encode(boxes, labels, anchors=self.anchors,
                             input_shape=self.input_shape, n_class=self.data.n_class)
            placed += encoded.placed
            unplaced += encoded.unplaced
        return TargetSurvey(placed=placed, unplaced=unplaced, dropped=dropped,
                            images=len(self))

    def annotations(self) -> list[Annotation]:
        """Every annotation in this split, for fitting anchors to it."""
        return [self.annotation(index) for index in range(len(self))]


def _empty_annotation(sample: Sample, height: int, width: int) -> Annotation:
    """A frame with no boxes, for a test split that has images and no labels."""
    return Annotation(
        path=Path(sample.images[0]), width=width, height=height,
        boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int), names=[],
        difficult=np.zeros(0, dtype=bool), image_path=str(sample.images[0]),
    )
