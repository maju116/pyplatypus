"""Many detectors, one spec - the detection counterpart of `Engine`.

The two engines are separate classes rather than one with a branch in every method,
because what they do differs at every step past discovery: three targets instead of one,
a loss that is part of the architecture, and an evaluation that cannot be accumulated per
batch because mean average precision is a property of a whole split's ranking.

`build_engine(spec)` returns whichever one the spec's task asks for, so a caller that
holds a spec never has to ask.

Three things here are decisions rather than plumbing:

* **anchors are fitted to the training annotations** when the spec does not give them, and
  recorded on the run. A detector cannot be reloaded without the anchors it was trained
  with - the same weights with different anchors predict boxes scaled by a fixed factor,
  with no other symptom.
* **predictions come back in the source image's own pixels**, as a list. There is no array
  form: images differ in size and so does the number of boxes found in each.
* **the target survey runs before training**, so a dataset the encoder cannot represent is
  visible in time to change `input_shape` rather than afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyplatypus.data.augmentation import build_box_augmenter
from pyplatypus.data.detection import DetectionDataset, TargetSurvey
from pyplatypus.data.paths import Sample, discover
from pyplatypus.detection.anchors import AnchorFit, anchor_coverage, box_shapes, generate_anchors
from pyplatypus.detection.boxes import non_max_suppression
from pyplatypus.detection.encode import COCO_ANCHORS, decode
from pyplatypus.detection.metrics import (
    COCO_THRESHOLDS,
    DetectionMetrics,
    detection_report,
    image_report,
)
from pyplatypus.detection.yolo3 import build_yolo3
from pyplatypus.engine import EngineBase, EngineError
from pyplatypus.spec.detection import DetectionModel
from pyplatypus.spec.spec import DetectionSpec, PlatypusSpec, SegmentationSpec
from pyplatypus.training.detection_trainer import DetectionTrainer
from pyplatypus.training.torch_data import make_detection_loader
from pyplatypus.training.trainer import History, seed_everything

#: Unplaced fraction above which the encoder is losing enough objects to matter. Measured
#: on BCCD at a 416 input: 3 boxes of 2804, 0.1%. A synthetic frame packed with 45 equal
#: boxes lost 9%, which is the shape of a dataset this would warn about.
_UNPLACED_WARNING = 0.02


@dataclass(frozen=True)
class DetectionReport:
    """One model on one split, under the three sets of conventions a table needs.

    Held as one object because they come from one pass over the data and because the
    conventions are part of the result: the same predictions score differently at IoU 0.5
    and averaged over 0.50-0.95, and precision and recall mean nothing without a
    confidence to read them at.
    """

    model: str
    split: str
    #: Average precision at IoU 0.5 - whether the objects were found.
    half: DetectionMetrics
    #: Averaged over IoU 0.50 to 0.95 in steps of 0.05, COCO's own - how well they fit.
    coco: DetectionMetrics
    #: The same predictions at `operating_point`, where precision and recall live.
    at_operating_point: DetectionMetrics

    def as_row(self, run: DetectorRun) -> dict[str, Any]:
        return {
            "model": self.model,
            "architecture": run.spec.architecture.value,
            "parameters": run.parameters,
            "epochs_run": len(run.history),
            "map_50": self.half.mean_average_precision,
            "map_50_95": self.coco.mean_average_precision,
            "mean_matched_iou": self.half.mean_matched_iou,
            "n_truth": sum(row["n_truth"] for row in self.half.per_class),
            "n_predicted": sum(row["n_predicted"] for row in self.half.per_class),
            "classes_without_truth": len(self.half.classes_without_truth),
        }

    def per_class(self) -> list[dict[str, Any]]:
        rows = []
        for row, at_point in zip(self.half.per_class,
                                 self.at_operating_point.per_class, strict=True):
            rows.append({
                "class": row["label"],
                "average_precision": row["average_precision"],
                "mean_matched_iou": row["mean_matched_iou"],
                "n_truth": row["n_truth"],
                "n_predicted": at_point["n_predicted"],
                "precision": at_point["precision"],
                "recall": at_point["recall"],
            })
        return rows


@dataclass
class DetectorRun:
    """Everything one detector left behind.

    `anchors` is here and not only in the spec because they may have been fitted, and in
    that case this is the only record of them.
    """

    name: str
    spec: DetectionModel
    model: torch.nn.Module
    trainer: DetectionTrainer
    anchors: tuple
    anchor_fit: AnchorFit | None = None
    survey: TargetSurvey | None = None
    history: History = field(default_factory=History)
    trained: bool = False

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class DetectionEngine(EngineBase):
    #: What a labelled split carries here, for the one message that names it.
    labels_are = "annotations"

    def __init__(self, spec: DetectionSpec, *, device: str | None = None,
                 num_workers: int = 0, strict_data: bool = True,
                 accumulate: int = 1):
        if not isinstance(spec, DetectionSpec):
            raise EngineError(
                f"this engine trains detectors; the specification's task is "
                f"'{spec.task.value}'. Use Engine for segmentation, or build_engine(spec) "
                f"and let the task choose."
            )
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.accumulate = accumulate
        self.runs: dict[str, DetectorRun] = {}
        seed_everything(spec.seed)

        self._samples: dict[str, tuple[Sample, ...]]
        self._discover_splits(discover, strict=strict_data)


    def _has_any_labels(self) -> bool:
        """Whether the test split carries annotations for anything, without reading them."""
        from pyplatypus.spec.common import DataMode

        path = Path(self.spec.data.test_path)
        if self.spec.data.mode is DataMode.CONFIG_FILE:
            if not path.is_file():
                return False
            first = path.read_text().splitlines()[:1]
            header = first[0] if first else ""
            return self.spec.data.label_column in [
                name.strip() for name in header.split(",")
            ]
        if not path.is_dir():
            return False
        wanted = self.spec.data.subdirs[1]
        return any((entry / wanted).is_dir() and any((entry / wanted).iterdir())
                   for entry in path.iterdir() if entry.is_dir())

    # ------------------------------------------------------------------ data
    def dataset(self, model: DetectionModel, split: str, *, anchors=None,
                only_images: bool = False, augmented: bool = False) -> DetectionDataset:
        if split not in self._samples:
            if split == "validation" and not self.spec.data.validation:
                raise EngineError(
                    "this run was specified with `validation: false`, so there is no "
                    "validation split to score. Name another split, or give "
                    "`validation_path` or `split` and fit again."
                )
            raise EngineError(
                f"no '{split}' data in this spec; available: "
                f"{', '.join(sorted(self._samples))}"
            )
        augmenter = build_box_augmenter(
            model.augmentation,
            input_shape=(int(model.input_shape[0]), int(model.input_shape[1])),
            min_visibility=model.min_visibility,
        ) if augmented else None
        return DetectionDataset(
            self._samples[split], model, self.spec.data,
            anchors=anchors if anchors is not None else self._anchors_for(model),
            only_images=only_images, augmenter=augmenter,
        )

    def loader(self, model: DetectionModel, split: str, *, anchors=None,
               shuffle: bool = False, augmented: bool = False):
        return make_detection_loader(
            self.dataset(model, split, anchors=anchors, augmented=augmented),
            batch_size=model.batch_size, shuffle=shuffle, num_workers=self.num_workers,
        )

    def _anchors_for(self, model: DetectionModel):
        """The anchors in use for this model: the run's if it has one, else the spec's.

        COCO's are the last resort and only reachable before `fit`, for a dataset whose
        anchors have not been fitted yet - looking at a target tensor needs *some* anchors.
        """
        run = self.runs.get(model.name)
        if run is not None:
            return run.anchors
        given = model.anchors_as_tuples
        return given if given is not None else COCO_ANCHORS

    # ------------------------------------------------------------------ anchors
    def fit_anchors(self, model: DetectionModel) -> AnchorFit | None:
        """k-means with an IoU distance over the training boxes, or nothing to do.

        Euclidean distance on (width, height) is scale-blind - (0.02, 0.02) and
        (0.04, 0.04) sit the same distance apart as (0.50, 0.50) and (0.52, 0.52), while
        their overlaps are 0.25 and 0.93 - so the distance is 1 - IoU, which is what the
        anchors are for.
        """
        if model.anchors_as_tuples is not None:
            return None
        dataset = self.dataset(model, "train", anchors=COCO_ANCHORS)
        return generate_anchors(
            dataset.annotations(), anchors_per_grid=model.anchors_per_grid,
            scales=3, input_shape=(int(model.input_shape[0]), int(model.input_shape[1])),
            seed=self.spec.seed if self.spec.seed is not None else 0,
        )

    def anchor_coverage(self, model_name: str, split: str = "train") -> dict[str, Any]:
        """How well the anchors in use cover this split's boxes.

        Worth asking of a split the anchors were *not* fitted on: anchors that cover the
        training boxes at 0.92 and the validation boxes at 0.70 say the two splits hold
        different objects, which no training curve shows.
        """
        run = self._run(model_name)
        self._needs_annotations(split)
        dataset = self.dataset(run.spec, split, anchors=run.anchors)
        shapes = box_shapes(dataset.annotations(),
                            input_shape=(int(run.spec.input_shape[0]),
                                         int(run.spec.input_shape[1])))
        flat = [pair for group in run.anchors for pair in group]
        return anchor_coverage(shapes, flat)

    def box_shapes(self, model_name: str, split: str = "train") -> dict[str, Any]:
        """The cloud of box shapes in a split, with the anchors in use beside it.

        Everything a picture of the anchor fit needs, in one call and in one set of
        coordinates: both are fractions of the model's input, computed by the same
        function that fitted the anchors. The alternative - widths and heights from one
        place and anchors from another - is how a plot comes to show boxes in different
        places from where the anchors were fitted, which looks like a bad fit and is a bug.

        Worth drawing for a split the anchors were *not* fitted on, for the same reason as
        `anchor_coverage`: it is the only way to see a class the anchors have nothing near.
        """
        from pyplatypus.detection.anchors import shape_table

        run = self._run(model_name)
        self._needs_annotations(split)
        dataset = self.dataset(run.spec, split, anchors=run.anchors)
        table = shape_table(
            dataset.annotations(), labels=list(self.spec.data.classes),
            input_shape=(int(run.spec.input_shape[0]), int(run.spec.input_shape[1])),
        )
        return {
            "boxes": table,
            "anchors": [[list(pair) for pair in group] for group in run.anchors],
            "anchors_were_fitted": run.anchor_fit is not None,
            "input_shape": [int(run.spec.input_shape[0]), int(run.spec.input_shape[1])],
            "classes": list(self.spec.data.classes),
        }

    # ------------------------------------------------------------------ fit
    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        for model_spec in self.spec.models:
            if verbose:
                print(f"\n=== {model_spec.name} "
                      f"({model_spec.architecture.value}) ===")

            if model_spec.weights:
                model_spec = self._adopt_from_weights(model_spec)

            network = build_yolo3(n_class=self.spec.data.n_class,
                                  anchors_per_grid=model_spec.anchors_per_grid,
                                  in_channels=model_spec.channels)

            if model_spec.weights:
                # The anchors come with the weights, because the weights only mean
                # anything with them. A specification that also named some is refused
                # while it is read, so there is nothing to reconcile here.
                sidecar = self._load_weights(network, model_spec.weights, model_spec)
                anchors = self._anchors_from(sidecar, model_spec)
                fit = None
                if verbose:
                    print(f"loaded '{model_spec.weights}' with its own anchors")
            else:
                fit = self.fit_anchors(model_spec) if model_spec.fit else None
                anchors = fit.anchors if fit is not None else self._anchors_for(model_spec)
                if verbose and fit is not None:
                    print(f"anchors fitted to {fit.boxes_used} boxes, mean IoU "
                          f"{fit.mean_iou:.4f} ({fit.boxes_dropped} dropped as degenerate)")

            trainer = DetectionTrainer(
                network, model_spec, anchors=anchors, n_class=self.spec.data.n_class,
                device=self.device, accumulate=self.accumulate,
            )
            run = DetectorRun(name=model_spec.name, spec=model_spec, model=trainer.model,
                              trainer=trainer, anchors=anchors, anchor_fit=fit)
            self.runs[model_spec.name] = run

            if model_spec.fit:
                run.survey = self._survey(run, verbose=verbose)
                run.history = trainer.fit(
                    # Augmented for training and never for validation: measuring a model
                    # on distorted data measures the distortion.
                    self.loader(model_spec, "train", anchors=anchors, augmented=True,
                                shuffle=self.spec.data.shuffle),
                    # None when the run says it has no validation set; `fit` has always
                    # taken an optional loader, so the history simply has no `val_` columns.
                    self.loader(model_spec, "validation", anchors=anchors)
                    if "validation" in self._samples else None,
                    verbose=verbose,
                )
                run.trained = True

            self._record(run)

        return {name: run.history for name, run in self.runs.items()}

    def _record(self, run: DetectorRun) -> None:
        """What this run leaves behind, when `output_dir` was asked for.

        The anchors are the point. When they were fitted rather than named, the
        specification that produced this detector does not contain them, and the same
        weights read with any others decode every box scaled by a fixed factor - so
        without this the only copy is the weights sidecar, and only if somebody
        remembered to export. `specification` plus `derived.anchors` re-runs it exactly.
        """
        from pyplatypus.runs import wants_a_record, write_record

        if not wants_a_record(self.spec):
            return
        derived = {
            "anchors": [[list(pair) for pair in group] for group in run.anchors],
            "anchors_were_fitted": run.anchor_fit is not None,
        }
        if run.anchor_fit is not None:
            derived["anchor_mean_iou"] = float(run.anchor_fit.mean_iou)
            derived["anchor_boxes_used"] = int(run.anchor_fit.boxes_used)
            derived["anchors_per_grid"] = run.spec.anchors_per_grid
        if run.survey is not None:
            derived["targets"] = run.survey.to_dict()
        write_record(self.spec, run.name, derived=derived, history=run.history)

    def _survey(self, run: DetectorRun, *, verbose: bool) -> TargetSurvey:
        """What the target can hold, before an epoch is spent."""
        survey = self.dataset(run.spec, "train", anchors=run.anchors).survey()
        if verbose:
            print(f"targets: {survey.placed} boxes placed, {survey.unplaced} could not be "
                  f"({survey.unplaced_fraction:.1%}), {survey.dropped} dropped as "
                  f"degenerate")
        if survey.unplaced_fraction > _UNPLACED_WARNING:
            import warnings

            warnings.warn(
                f"model '{run.spec.name}': the target cannot hold "
                f"{survey.unplaced_fraction:.1%} of the training boxes "
                f"({survey.unplaced} of {survey.total}). Two objects of one shape whose "
                f"centres land in the same grid cell share a slot, so the second is "
                f"dropped - it is never shown to the model and never counted as missed. "
                f"A larger input_shape gives finer grids; more anchors per grid gives "
                f"more slots in each.",
                stacklevel=3,
            )
        return survey

    def _load_weights(self, model: torch.nn.Module, reference: str,
                      spec: DetectionModel) -> dict | None:
        """Load, checking the two things the model specification cannot answer itself.

        `n_class` sets the head's width and lives on the data; the class *names* decide
        what every predicted label means. Weights trained on three classes in another
        order load without complaint and label every box wrongly - the detection
        counterpart of §4n's weights trained on a different colormap.
        """
        from pyplatypus.weights import load_into

        return load_into(model, reference, spec, extra={
            "n_class": self.spec.data.n_class,
            "classes": list(self.spec.data.classes),
        })

    #: `anchors_per_grid` decides the shape of every head, and a published detector knows
    #: its own. Not `input_shape` or `channels`, for the reason the segmentation engine
    #: gives: the rank is needed while the specification is validated, before a sidecar can
    #: be reached without a download.
    ADOPTABLE = ("architecture", "anchors_per_grid")

    def _adopt_from_weights(self, model_spec: DetectionModel) -> DetectionModel:
        """The counterpart of `_anchors_from` for the rest of what the file describes.

        Anchors have travelled with detection weights since 0.3.0a12, for the reason that
        without them the same weights decode every box scaled by a fixed factor. The head
        geometry is the same kind of fact, and a caller should no more have to know a
        published detector's `anchors_per_grid` than its anchors.
        """
        from pyplatypus.weights import describe, resolve_weights

        try:
            sidecar = describe(resolve_weights(model_spec.weights))
        except Exception:  # noqa: BLE001 - a bad reference is load_into's story to tell,
            return model_spec          # told with the message it has always given.
        if not sidecar:
            return model_spec
        adopted = {field: sidecar[field] for field in self.ADOPTABLE
                   if field in sidecar and field not in model_spec.model_fields_set}
        if not adopted:
            return model_spec
        return type(model_spec).model_validate({**model_spec.model_dump(), **adopted})

    def _anchors_from(self, sidecar: dict | None, spec: DetectionModel) -> tuple:
        """The anchors the weights were trained with, or a refusal.

        A detector without its anchors cannot be used at all: every box it decodes would
        be scaled by whatever factor separates the anchors it learned from the ones it is
        read with. There is no sensible default, so a file that does not carry them is
        refused rather than guessed at.
        """
        recorded = (sidecar or {}).get("anchors")
        if not recorded:
            raise EngineError(
                f"'{spec.weights}' carries no anchors, so the boxes it predicts cannot be "
                f"decoded. A detector's weights are relative to the anchors they were "
                f"trained with and nothing can recover them from the weights themselves. "
                f"Weights written by DetectionEngine.export_weights record them in the "
                f"sidecar beside the file; one converted from elsewhere needs them added."
            )
        anchors = tuple(tuple((float(w), float(h)) for w, h in group)
                        for group in recorded)
        widths = {len(group) for group in anchors}
        if widths != {spec.anchors_per_grid}:
            raise EngineError(
                f"'{spec.weights}' was trained with {sorted(widths)} anchors per grid and "
                f"the model asks for {spec.anchors_per_grid}; the head is one tensor of "
                f"width anchors_per_grid * (n_class + 5), so these weights do not fit it."
            )
        return anchors

    # ------------------------------------------------------------------ predict
    def predict(self, model_name: str, split: str = "test") -> list[dict[str, Any]]:
        """Boxes, scores and labels for every image in a split, in its own pixels.

        A list and not an array, and in source pixels rather than the network's frame -
        both for the same reason the segmentation side returns a list for
        `space="source"`. Images differ in size, the number of boxes found differs per
        image, and a box in a letterboxed 416x416 frame cannot be drawn on the photograph
        it came from without undoing the letterbox, which is a step nobody should have to
        remember.
        """
        run = self._run(model_name)
        dataset = self.dataset(run.spec, split, anchors=run.anchors,
                               only_images=split not in self.labelled)
        from pyplatypus.training.torch_data import to_channels_first

        out = []
        for index in range(len(dataset)):
            example = dataset.read(index)
            outputs = run.trainer.raw_outputs(to_channels_first(example.image))
            boxes, scores, labels = decode(
                outputs, anchors=run.anchors,
                input_shape=(int(run.spec.input_shape[0]), int(run.spec.input_shape[1])),
                n_class=self.spec.data.n_class,
                objectness=run.spec.score_threshold, raw=True,
            )
            keep = non_max_suppression(boxes, scores, labels,
                                       iou_threshold=run.spec.nms_threshold)
            out.append({
                "key": dataset.samples[index].key,
                "boxes": example.fit.inverse(boxes[keep]) if len(keep)
                         else np.zeros((0, 4)),
                "scores": scores[keep],
                "labels": labels[keep],
                "names": [self.spec.data.classes[i] for i in labels[keep]],
            })
        return out

    def crops(self, model_name: str, split: str = "test", *,
              score_threshold: float | None = None, context: float = 0.0,
              size: tuple[int, int] | None = None, fit: str = "letterbox",
              fill: float = 0.5) -> list[dict[str, Any]]:
        """Every detection cut out of the image it was found in.

        The pipeline this is for is a detector followed by a classifier it does not
        contain: find the objects with three classes, crop them, hand the crops to
        something that knows eighty. It is the second thing people ask for after boxes,
        and doing it by hand means re-reading every image and getting the rounding right.

        **Filtered at `operating_point`, not at `score_threshold`.** The specification's
        `score_threshold` is deliberately near zero so that average precision integrates
        the whole ranking - on BCCD that is hundreds of boxes an image, almost all of them
        the tail that AP exists to measure over. Cropping them would hand a classifier
        mostly noise. `operating_point` is where someone stands when they act on a
        prediction, which is what cropping is.

        One record per image rather than one flat list, because a crop without its source
        is not traceable: `key` says which image, and `boxes` are the coordinates it came
        from, in that image's own pixels.

        Args:
            model_name: which trained detector.
            split: which split's images to read.
            score_threshold: override the confidence to crop at.
            context: expand each box by this fraction of its size per side, 0 by default.
            size: bring every crop to `(height, width)`, or None to keep each as cut.
            fit: `"letterbox"` to preserve aspect, `"stretch"` to resize both axes.
            fill: padding value for `"letterbox"`.

        Returns:
            One dict per image with `key`, `crops`, `boxes`, `scores`, `labels`, `names`.
        """
        from pyplatypus.detection.boxes import clip_boxes, crop_boxes, drop_degenerate

        run = self._run(model_name)
        cut = run.spec.operating_point if score_threshold is None else float(score_threshold)
        dataset = self.dataset(run.spec, split, anchors=run.anchors,
                               only_images=split not in self.labelled)

        # One pass over the sample list rather than a scan per image: a split with a
        # thousand images would otherwise be a million comparisons to find each one.
        by_key = {sample.key: i for i, sample in enumerate(dataset.samples)}

        out = []
        for found in self.predict(model_name, split):
            image = dataset.source_image(by_key[found["key"]])
            keep = np.flatnonzero(found["scores"] >= cut)

            # Clipped and then dropped, which is what `DetectionDataset.read` already does
            # to the truths it reads. A model is free to predict a box off the frame and
            # `predict` does not clip, so at a low threshold some come back with no overlap
            # at all - a zero-extent box at the edge once the letterbox is undone. There is
            # nothing to crop there, and `dropped` says how many rather than the count
            # quietly not matching the boxes.
            boxes = clip_boxes(found["boxes"][keep], image.shape[:2])
            boxes, kept_index, dropped = drop_degenerate(boxes, keep)

            out.append({
                "key": found["key"],
                "crops": crop_boxes(image, boxes, context=context, size=size, fit=fit,
                                    fill=fill),
                "boxes": boxes,
                "scores": found["scores"][kept_index],
                "labels": found["labels"][kept_index],
                "names": [found["names"][i] for i in kept_index],
                "dropped": dropped,
            })
        return out

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, split: str = "validation") -> list[dict[str, Any]]:
        """One row per model: the comparison table, with detection's columns.

        Different columns from the segmentation table, because they answer a different
        question. `mean_matched_iou` is the one worth keeping beside the averages: the gap
        between mAP@0.5 and mAP@[.50:.95] is localisation, and this is that gap as a
        single number - how well the boxes that matched actually fit, rather than merely
        that they cleared a threshold.

        **No precision or recall here**, although `operating_point` says where to read
        them. Averaging them over classes needs a weighting and every choice of weighting
        is a different claim: summed over BCCD's 4155 red cells, 372 white and 361
        platelets, a single precision is a statement about red cells wearing the costume
        of a statement about the model. They are in `evaluate_classes`, per class, which
        is the only form in which they mean anything. `DetectionMetrics.as_rows` made the
        same decision - its "all" row leaves both as None.
        """
        if not self.runs:
            raise EngineError("nothing has been trained or loaded yet; call fit() first")
        return [self.report(name, split).as_row(run)
                for name, run in self.runs.items()]

    def evaluate_classes(self, model_name: str, split: str = "validation"
                         ) -> list[dict[str, Any]]:
        """One row per class instead of one row per model.

        The row that matters on an unbalanced dataset, which is most of them: BCCD has
        4155 red cells against 372 white and 361 platelets, so a single number is a number
        about red cells.
        """
        return self.report(model_name, split).per_class()

    def evaluate_images(self, model_name: str, split: str = "validation",
                        score_threshold: float | None = None
                        ) -> list[dict[str, Any]]:
        """One row per image instead of one row per model or per class.

        The question a table of averages cannot answer: *which* images it fails on. The
        counterpart of the segmentation engine's per-case scores, and the second question
        anyone asks after seeing a mean.

        **No average precision per image.** AP is the area under a precision-recall curve
        and therefore a property of a ranking over a dataset; on one image with three boxes
        it swings on a single box's rank and says nothing. The columns are counts and
        overlap, which do mean something for one picture.

        **Counts are read at the specification's `operating_point`** unless another
        `score_threshold` is given, because "how many were missed" is undefined over the
        whole ranking - at a threshold of zero every box the model dimly considered is a
        prediction, and `spurious` would count the tail that average precision exists to
        integrate over rather than anything a user would see.

        Sort by `missed` to find the frames it cannot see, or by `mean_matched_iou` to find
        the ones where it sees everything and places it badly. Those are different
        problems: the first is usually the data, the second is usually the anchors.
        """
        run = self._run(model_name)
        self._needs_annotations(split, model_name=run.name)
        predictions = self.predict(model_name, split)
        dataset = self.dataset(run.spec, split, anchors=run.anchors)
        truths = [dataset.annotation(index).as_truth() for index in range(len(dataset))]
        point = (run.spec.operating_point if score_threshold is None
                 else float(score_threshold))
        return image_report(
            predictions, truths,
            labels=list(self.spec.data.classes),
            score_threshold=point,
            keys=[sample.key for sample in dataset.samples],
        )

    def split_sizes(self) -> dict[str, int]:
        """How many images each split holds, and in which order they were found."""
        return {name: len(samples) for name, samples in self._samples.items()}

    def report(self, model_name: str, split: str = "validation") -> DetectionReport:
        """Everything a detection table is read off, from one pass over the split.

        Public because `evaluate` and `evaluate_classes` are both views over it, and a
        caller that wants both - which the example does - would otherwise run the model
        over the split twice. No caching: a cache would be stale the moment the model
        trained another epoch, and the honest alternative is to hand back the thing and
        let the caller hold it.
        """
        half, coco, point = self._reports(model_name, split)
        return DetectionReport(model=model_name, split=split, half=half, coco=coco,
                               at_operating_point=point)

    def _reports(self, model_name: str, split: str):
        """The three reports every detection table is read off.

        Two thresholds and they are not the same thing. `score_threshold` is kept low
        because average precision is a property of the **whole ranking** - cutting the
        tail off removes the part of the curve AP integrates over and inflates the score.
        `operating_point` is where precision and recall are read, because those are a
        single choice of confidence and mean nothing without one.
        """
        run = self._run(model_name)
        self._needs_annotations(split, model_name=run.name)
        predictions = self.predict(model_name, split)
        dataset = self.dataset(run.spec, split, anchors=run.anchors)
        truths = [dataset.annotation(index).as_truth() for index in range(len(dataset))]
        classes = list(self.spec.data.classes)
        return (
            detection_report(predictions, truths, labels=classes,
                             iou_thresholds=(0.5,), interpolation="101"),
            detection_report(predictions, truths, labels=classes,
                             iou_thresholds=COCO_THRESHOLDS, interpolation="101"),
            detection_report(predictions, truths, labels=classes,
                             iou_thresholds=(0.5,), interpolation="101",
                             score_threshold=run.spec.operating_point),
        )

    # ------------------------------------------------------------------ odds and ends
    def _needs_annotations(self, split: str, model_name: str = "<model>") -> None:
        """What this engine calls a labelled split, and what to do instead."""
        self._needs_labels(
            split,
            f"predict('{model_name}', '{split}') works on it.",
        )

    def _run(self, model_name: str) -> DetectorRun:
        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        return run

    def model_names(self) -> list[str]:
        return list(self.runs)

    def export_weights(self, model_name: str, path: str | Path, **extra) -> Path:
        """Write one detector's weights, with the anchors in the sidecar.

        The anchors are not optional metadata. The same weights read with different
        anchors decode every box scaled by a fixed factor, and nothing about the output
        says so - the boxes are plausible, the scores are plausible, and they are in the
        wrong places.
        """
        from pyplatypus.weights import export_weights

        run = self._run(model_name)
        payload = {
            "anchors": [[list(pair) for pair in group] for group in run.anchors],
            "classes": list(self.spec.data.classes),
            **extra,
        }
        return export_weights(run.model, run.spec, path, extra=payload)

    def best_model(self, key: str = "map_50", split: str = "validation") -> str:
        table = self.evaluate(split)
        if key not in table[0]:
            available = ", ".join(k for k, v in table[0].items()
                                  if isinstance(v, (int, float)))
            raise EngineError(
                f"no column '{key}' in the evaluation table; available: {available}"
            )
        if any(row[key] is None for row in table):
            raise EngineError(
                f"'{key}' is undefined for at least one model, so they cannot be ranked "
                f"on it. A class with no truth boxes in this split has no average "
                f"precision, and a mean over classes that excludes it is not comparable."
            )
        return max(table, key=lambda row: row[key])["model"]


def build_engine(spec: PlatypusSpec, **kwargs):
    """The engine this spec's task asks for.

    Here rather than in `engine.py` to keep the import one way round: detection imports
    `EngineError` from there and nothing comes back.
    """
    from pyplatypus.engine import Engine

    if isinstance(spec, DetectionSpec):
        return DetectionEngine(spec, **kwargs)
    if isinstance(spec, SegmentationSpec):
        return Engine(spec, **kwargs)
    raise EngineError(
        f"no engine for task '{getattr(spec, 'task', '?')}'"
    )
