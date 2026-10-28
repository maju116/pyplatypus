"""Many models, one spec.

This is what `platypus_fit(spec)` calls. Each model gets its own data pipeline, because
each may want a different input size or a different tiling, and its own augmentation,
because that is part of the experiment. Validation never gets augmented - measuring a
model on distorted data measures the distortion.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from pyplatypus.data.augmentation import build_augmenter
from pyplatypus.data.dataset import SegmentationDataset, _is_series
from pyplatypus.data.paths import Sample, discover
from pyplatypus.data.splits import group_of
from pyplatypus.errors import PlatypusError
from pyplatypus.models import build_model
from pyplatypus.models.encoders import PretrainedEncoder
from pyplatypus.objectives.losses import loss_needs_distance, loss_needs_mask
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.spec.spec import PlatypusSpec, SegmentationSpec
from pyplatypus.training.torch_data import make_loader
from pyplatypus.training.trainer import History, Trainer, seed_everything

#: Unmatched fraction above which the colormap is probably wrong. Measured: correct masks
#: give 0.00%, JPEG compression around a mask's edge gives 0.72%, a wrong foreground colour
#: gives the size of the foreground (19.8% measured on a 20% object), a wrong background
#: gives 100%. This sits in the gap, nearer the artefact end.
_UNMATCHED_WARNING = 0.05


class EngineError(PlatypusError):
    kind = "engine_error"


class EngineBase:
    """What both engines do with a specification before either becomes itself.

    A narrow base, and the width was measured rather than chosen. Across the two engines
    twelve method names are shared, and only two of them are the same work: `__init__` at
    0.70 similarity and `_discover_test` at 0.65, the latter differing in nothing but its
    docstring and the name of one hook. Everything else is lower - `predict` is 0.05 and
    equal in length by coincidence - so pulling more up would produce methods that are
    mostly a branch on which subclass is calling them.

    The argument for doing even this much is not tidiness. These two have already diverged
    twice: detection recorded which splits carry annotations while segmentation threw the
    same fact away, and the guard order that made "no masks" mask "no split" was wrong in
    both and had to be fixed in both. The second time, the fix was written twice on the
    same night.

    It lives in this module because `EngineError` and `split_from_train` already do, and
    both engines already import from here. A third file for three shared things would be a
    third place to look.

    A subclass provides `_has_any_labels()`, which is the one question `_discover_test`
    cannot answer for itself: masks and annotations are found in different ways.
    """

    #: What a labelled split carries, for the one message that has to name it.
    labels_are = "labels"

    def _resolve_model(self, model_name: str | None) -> str:
        """The model a method was asked about, with no name meaning the first one.

        Nine docstrings across the two engines said `model_name` "defaults to the first"
        and not one of them had a default, so nine published pages invited a reader to omit
        an argument that raised `TypeError`. The resolution lives here rather than in each
        body so the two engines cannot come to answer it differently - which is the thing
        this class exists for.
        """
        if model_name is not None:
            return model_name
        first = next(iter(self.runs), None)
        if first is None:
            raise EngineError(
                "no model has been trained or loaded yet, so there is no first one to use; "
                "call `fit` first, or name a model"
            )
        return first

    def _discover_splits(self, discover, *, strict: bool) -> None:
        """`self._samples` and `self.labelled`, from the data block alone.

        Training is always there. Validation is there unless the run said it has none, and
        **absent rather than empty** in that case: an empty split makes every count read
        zero and every average read nan, which is a number for something nobody asked for.
        A test split may or may not carry labels, and `_discover_test` decides which.
        """
        data = self.spec.data

        self.labelled = {"train"}
        if data.validation:
            self.labelled.add("validation")

        if data.split is not None:
            self._samples = split_from_train(data, discover, strict=strict)
            # A split cut out of the training folder came from labelled data, so unlike a
            # separate `test_path` its test fraction can be scored and not only predicted on.
            if "test" in self._samples:
                self.labelled.add("test")
            return

        self._samples = {
            "train": discover(data.train_path, data, strict=strict).samples,
        }
        if data.validation:
            self._samples["validation"] = discover(
                data.validation_path, data, strict=strict
            ).samples
        if data.test_path:
            self._samples["test"] = self._discover_test(discover, strict=strict)

    def _discover_test(self, discover, *, strict: bool) -> tuple[Sample, ...]:
        """The test split, with its labels if it has any.

        Labels are asked for first and their absence is not an error - a test set of images
        alone is an ordinary thing to have, and `predict` is what it is for. The split is
        then recorded as unlabelled, so `evaluate` refuses it by name rather than scoring
        it against nothing.

        **"None" and "some" are not the same thing.** Falling back whenever labelled
        discovery failed would turn a test split with two missing files into an unlabelled
        one, throwing away the seventy that are there and reporting nothing. So the fallback
        happens only when the split carries no labels at all; an incomplete one raises,
        which is what `strict` is for. Met on the detection side first.
        """
        from pyplatypus.errors import ConfigError

        data = self.spec.data
        try:
            samples = discover(data.test_path, data, strict=strict).samples
        except ConfigError:
            if self._has_any_labels():
                raise
            return discover(data.test_path, data, only_images=True, strict=strict).samples
        self.labelled.add("test")
        return samples

    def _has_any_labels(self) -> bool:  # pragma: no cover - a subclass answers this
        """Whether the test split carries labels for anything, without reading them."""
        raise NotImplementedError

    def _require_split(self, split: str) -> None:
        """Say what is missing, here, rather than letting a KeyError say it two layers down.

        `validation: false` is the common way to arrive: `evaluate` and its neighbours
        default to `"validation"`, so a run that deliberately has none meets this at the
        first thing anyone calls after `fit`.
        """
        if split in self._samples:
            return
        if split == "validation" and not self.spec.data.validation:
            raise EngineError(
                "this run was specified with `validation: false`, so there is no validation "
                "split to score. Name another split, or give `validation_path` or `split` "
                "and fit again."
            )
        raise EngineError(
            f"no '{split}' data in this spec; available: {', '.join(sorted(self._samples))}"
        )

    def _needs_labels(self, split: str, suggestion: str) -> None:
        """Refuse a question about labels that are not there.

        Scoring an unlabelled split would either fail deep in the mask layer - which is what
        it used to do - or, worse, report a number. A zero in a table is read as a result.

        Order matters: a split that was never created is a different thing from one that
        exists without labels, and "no labels" sends somebody looking for files when what
        they wrote was `validation: false`.
        """
        self._require_split(split)
        if split not in self.labelled:
            raise EngineError(
                f"the '{split}' split has no {self.labels_are}, so there is nothing to "
                f"score against. {suggestion}"
            )

    #: Which of a model's fields a sidecar may fill in where the specification stayed
    #: silent, as `field: the key the sidecar records it under`. Empty here and declared
    #: by each engine below, because what a published file knows about itself is
    #: task-specific: a detector's head is set by its anchor count and a U-shaped one by
    #: blocks and filters.
    #:
    #: **A mapping rather than a list of names**, because the two differ exactly once:
    #: the instance engine adopts `separation` from `separation_settings`, the sidecar
    #: having to call it something else - `separation` there is the signal alone and part
    #: of what the weights say they are *for*, so a block written under that name would
    #: shadow the fingerprint and make the file refuse to load into the model it came
    #: from. `pyplatypus#236`.
    ADOPTABLE: ClassVar[dict[str, str]] = {}

    def _adopt_from_weights(self, model_spec):
        """Fill what the weights already describe, where the specification stayed silent.

        A published file knows its own geometry - `dsbowl-unet` records u_net, 4 blocks, 16
        filters - so requiring the caller to restate it means knowing the internals of
        somebody else's model in order to use it, and being refused for guessing wrong.
        Anchors have travelled with detection weights this way since 0.3.0a12, for the
        sharper version of the same reason: without them the same weights decode every box
        scaled by a fixed factor.

        `model_fields_set` is what makes this safe: it tells a value the user chose from a
        default that happened to be there. Anything stated is left alone and still has to
        agree, so this loosens what may be omitted and nothing about what is checked.

        Returns:
            The specification, and what was taken from the file - so a caller can say so.
            Both engines that had their own copy of this ignore the second half; the
            instance engine prints it, because what it adopts includes a training decision
            and the argument for adopting one at all is that the reader still sees it.

        **One mechanism, three declarations, as of `0.8.0a28`.** This body existed twice,
        character for character apart from a type hint, in `Engine` and `DetectionEngine`;
        `InstanceEngine` was about to need a third with one asymmetric entry. That is the
        shape `#236` is about, so the copies were merged rather than added to.
        """
        from pyplatypus.weights import describe, resolve_weights

        try:
            sidecar = describe(resolve_weights(model_spec.weights))
        except Exception:  # noqa: BLE001 - a bad reference is load_into's story to tell,
            return model_spec, {}  # told with the message it has always given.
        if not sidecar:
            return model_spec, {}

        adopted = {
            field: sidecar[key]
            for field, key in self.ADOPTABLE.items()
            if key in sidecar and field not in model_spec.model_fields_set
        }
        if not adopted:
            return model_spec, {}
        # Re-validated rather than copied, because the adopted values have to face the same
        # checks a written one would: `blocks` decides what input sizes are divisible, and
        # an adopted `separation` has to be a block this build of the engine can read.
        return type(model_spec).model_validate({**model_spec.model_dump(), **adopted}), adopted


@dataclass
class ModelRun:
    """Everything one model left behind."""

    name: str
    spec: SegmentationModel
    model: torch.nn.Module
    trainer: Trainer
    history: History = field(default_factory=History)
    trained: bool = False

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class Engine(EngineBase):
    #: What a labelled split carries here, for the one message that names it.
    labels_are = "masks"

    def __init__(
        self,
        spec: PlatypusSpec,
        *,
        device: str | None = None,
        num_workers: int = 0,
        strict_data: bool = True,
        check_masks: bool = True,
    ):
        if not isinstance(spec, SegmentationSpec):
            # Named per task rather than as one sentence about "not segmentation": the two
            # other tasks are at different stages, and a message that lumped them together
            # would send an instance specification to the detection examples.
            elsewhere = {
                "object_detection": (
                    "The detection half of the spec exists and validates, but nothing "
                    "trains from it yet - until it does, a detector is assembled by hand "
                    "from pyplatypus.detection, the way examples/detect_blood_cells.py does."
                ),
                "instance_segmentation": (
                    "`InstanceEngine` trains that one - the head is wider than the class "
                    "count and the score is computed after a post-process, which is why it "
                    "is a different engine rather than a flag on this one. "
                    "`build_engine(spec)` picks by the tag, so holding a specification is "
                    "enough."
                ),
            }
            raise EngineError(
                f"this engine trains semantic segmentation; the specification's task is "
                f"'{spec.task.value}'. " + elsewhere.get(spec.task.value, "")
            )
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.check_masks = check_masks
        self.runs: dict[str, ModelRun] = {}
        seed_everything(spec.seed, deterministic=spec.deterministic)

        self._samples: dict[str, tuple[Sample, ...]]
        self._discover_splits(discover, strict=strict_data)

    def _has_any_labels(self) -> bool:
        """Whether the test split carries masks for anything, without reading them."""
        return labels_in_test_split(self.spec.data)

    def dataset(
        self,
        model: SegmentationModel,
        split: str,
        *,
        augmented: bool = False,
        only_images: bool | None = None,
    ) -> SegmentationDataset:
        """`only_images` defaults to what the split actually has rather than to False.

        It used to default to False, so asking for a test split of images alone built a
        dataset that would read masks and fail on the first item. The engine knew: it had
        just discovered the split without them.

        Args:
            model: Which model's settings to read.
            split: Which data.
            augmented: Apply the model's augmentation. Training only.
            only_images: Read images alone. Defaults to what the split actually has.

        Returns:
            A `SegmentationDataset`. `inspect_masks()` on it is what `fit` calls to refuse
            a colormap matching none of the labelled tissue.
        """
        self._require_split(split)
        if only_images is None:
            only_images = split not in self.labelled
        augmenter = (
            build_augmenter(model.augmentation, model.rank, tuple(model.input_shape))
            if augmented
            else None
        )
        return SegmentationDataset(
            self._samples[split],
            model,
            self.spec.data,
            augmenter=augmenter,
            only_images=only_images,
        )

    def loader(
        self,
        model: SegmentationModel,
        split: str,
        *,
        augmented: bool = False,
        shuffle: bool = False,
        only_images: bool = False,
        for_loss: bool = True,
    ):
        """A torch DataLoader over one split, wrapping `dataset`.

        Args:
            model: Which model's settings to read - `input_shape`, `batch_size` and the
                augmentation all come from here.
            split: `"train"`, `"validation"` or `"test"`.
            augmented: Apply the model's augmentation. Training only: validation is read
                the same way every epoch, or two epochs measure different things.
            shuffle: Shuffle between epochs. Training only, for the same reason.
            only_images: Read images alone, with no targets.
            for_loss: Whether these batches are going to the loss. When they are and the
                loss asks for one, each batch carries a third item: the signed distance map
                for `boundary`, or the foreground mask for a regression loss scoring the
                foreground. Both are computed in the loader's workers rather than in the
                loss - the distance transform costs more than an epoch of a small 3D model -
                and the paths that only score or predict should not pay for either.

        Returns:
            A `torch.utils.data.DataLoader`. Batches hold two items, or three when the
            loss asked the loader for something.
        """
        # `boundary` needs the signed distance map of its target and nothing else does, so
        # the loader computes it only when the loss asks - the transform costs more than an
        # epoch of a small 3D model and nobody else should pay for it. Read from the spec
        # rather than from the trainer's loss object, because the loader is built first.
        #
        # `for_loss=False` is for the paths that score or predict rather than optimise:
        # they never call the loss, so a map built for them would be pure waste - 700 ms a
        # sample at 128^3, on work that does not read it.
        #
        # A regression loss asks for the other extra tensor, the mask of the voxels its
        # target is defined on. Same question, same place, same reason it is asked of the
        # spec; the two are never both true, and `TorchSegmentationDataset` refuses the
        # combination rather than silently serving one of them.
        #
        # **Since `0.8.0a21` no loss a semantic specification can name answers yes**, because
        # a model's own `loss` scores the classes and the three continuous ones are refused
        # there (`pyplatypus#223`). The branch stays rather than becoming `False`: the
        # question belongs to the loss, this is the one place it is asked, and hard-coding
        # the consequence would be a second place where "a semantic loss is never
        # continuous" is written down. `InstanceEngine.loader` asks its own version - the
        # signal, not the loss, because a flow block's objective is not the model's.
        # `test_no_loss_a_semantic_model_can_name_asks_for_the_foreground` holds it.
        return make_loader(
            self.dataset(model, split, augmented=augmented, only_images=only_images),
            batch_size=model.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            with_distance=for_loss and loss_needs_distance(model.loss),
            with_foreground=for_loss and loss_needs_mask(model.loss),
        )

    # ----------------------------------------------------------------- weights
    def _load_weights(
        self, model: torch.nn.Module, reference: str, spec: SegmentationModel
    ) -> dict | None:
        """A registry name, a Hub reference, or a local path - see `pyplatypus.weights`.

        The colormap goes into the comparison, and it is the thing the model specification
        cannot answer for itself: it lives on the data. Without it, weights trained to call
        `(255, 0, 0)` class 1 load cleanly into a model whose class 1 is `(0, 255, 0)`, and
        every mask comes back confidently wrong - which `pyplatypus.weights` has described
        in prose since it was written while recording nothing that could catch it. The
        detection side already compared its class *names*; this is the same check.
        """
        from pyplatypus.weights import load_into

        return load_into(model, reference, spec, extra=self._class_fingerprint())

    def _class_fingerprint(self) -> dict:
        """What the data says the classes are. Exactly one of the three namings is set.

        `n_class` is here rather than on the model because the data decides it. It used to
        be a model field, which made it a setting that could disagree with its own data and
        needed a validator to say so; now there is nothing to disagree with. The count is
        still compared when weights are loaded - it is simply contributed by the half of the
        specification that knows it.

        The naming itself goes in because this dictionary is written into every run record
        and every weights sidecar, and `n_class` alone cannot tell the two worlds apart:
        three exclusive classes and three overlapping regions are the same count and not the
        same model, so weights from one must not be read back as the other.
        """
        data = self.spec.data
        classes = {"n_class": data.n_class}
        if data.regions is not None:
            return {
                **classes,
                "regions": [{"name": r.name, "of": list(r.of)} for r in data.regions],
            }
        if data.classes_from is not None:
            # The patterns are the class identity here, the way detection records its class
            # *names*: three independent classes in a different order are the same count and
            # the same shape, and weights read back that way would label every channel
            # wrongly while answering plausibly.
            return {**classes, "classes_from": list(data.classes_from)}
        if data.labels is not None:
            return {**classes, "labels": list(data.labels)}
        return {**classes, "colormap": [list(colour) for colour in data.colormap]}

    def export_weights(self, model_name: str, path: str | Path, **extra) -> Path:
        """Write one trained model's weights, ready to publish or to load again later.

        safetensors plus a sidecar recording what the weights are for. Anything passed as
        `extra` joins the sidecar, which is where the data they were trained on and its licence
        belong - a weights file whose provenance is only in somebody's memory cannot be used by
        anybody else.

        Args:
            model_name: Which model's weights to write.
            path: Where to write. The sidecar goes beside it.
            **extra: Anything else to record in the sidecar.

        Returns:
            The path written.
        """
        from pyplatypus.weights import export_weights

        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        # The colormap always, whatever else the caller adds: it is what makes the sidecar
        # able to refuse weights that would load cleanly and mean something else.
        return export_weights(
            run.model, run.spec, path, extra={**self._class_fingerprint(), **(extra or {})}
        )

    # ------------------------------------------------------------------ checks
    def _refuse_masks_that_describe_nothing(self, model_spec: SegmentationModel) -> None:
        """Look at the training masks before training on them.

        A colormap or a list of labels that does not match the masks is the most common
        way a run trains on nothing at all, and it is silent: the loss falls, the metrics
        look plausible because the background is most of every medical image, and the
        model learns to answer "background" everywhere. Nothing in a training log says so.

        A **declared class that appears in no mask** is refused. That is provable rather
        than suspicious - its channel in the target is zero everywhere, so there is no
        gradient towards it and the model is never shown the thing it is being asked to
        find. It is also the one signal that does not shrink with the lesion: see
        `SegmentationDataset.inspect_masks` for the measurement.

        A high unmatched fraction with every class present is only **warned** about,
        because some of it is legitimate - JPEG compression around the edge of a mask
        measured 0.72% on Kvasir-SEG, and antialiasing does the same. The threshold sits
        in the gap between that and a wholesale mistake, which measured 19.8% for a wrong
        foreground colour and 100% for a wrong background.

        **`labels` without 0 and any unmatched voxel at all is refused**, which is the one
        place the threshold above is wrong rather than loose (`platypus#153`). Index 0 means
        two things here and always has: the first entry of a naming, and *matched nothing*.
        Everywhere else that ambiguity is harmless because `labels: [0, ...]` makes them
        agree. Take the 0 away and every unmatched voxel - the background, usually most of
        the image - becomes the **first declared class**, so the model is trained to call
        background a nucleus with nothing in the shape of the target to give it away: the
        target is in range, one-hot and plausible.

        Refused at *any* unmatched voxel rather than at the warning threshold, because this
        is a label map and not a picture. Label maps hold integers, are read
        nearest-neighbour and compared with half a unit of tolerance, so the legitimate case
        - `labels: [1, 2]` over masks that really do fill every voxel - produces an unmatched
        fraction of **exactly zero**, and that case keeps working. The compression and
        antialiasing that justify the 5% threshold belong to colormaps, which cannot hit
        this check: a naming without 0 is only expressible in `labels`.
        """
        if not self.check_masks:
            return
        # Nearest-neighbour reading of a mask is cheap in 2D and not in 3D, where every
        # sample is a volume. Fewer samples there, enough that absence still means
        # something.
        limit = 8 if model_spec.rank == 3 else 50
        dataset = self.dataset(model_spec, "train")
        report = dataset.inspect_masks(limit=limit)

        labels = self.spec.data.labels
        if labels is not None and 0 not in labels and report.unmatched > 0:
            raise EngineError(
                f"labels={labels} does not contain 0, and {report.unmatched:.1%} of the "
                f"training masks' voxels match none of those labels - checked "
                f"{report.samples_checked} of {report.total_samples}.\n"
                f"  Index 0 is the class a voxel matching nothing falls back to, so those "
                f"voxels are being trained as '{labels[0]}' - the first label - and the "
                f"target stays in range, one-hot and plausible while it happens.\n"
                f"  If those voxels are background, add 0 to labels: "
                f"{[0, *labels]}, which is one more class and one more channel.\n"
                f"  If they are not background, something else does not describe these "
                f"masks: labels that do fill every voxel give exactly 0% unmatched.\n"
                f"  To see what the values actually are:\n"
                f"    engine.dataset(spec.models[0], 'train').inspect_masks(limit=500)"
            )

        if report.missing_classes:
            described = self._describe_classes(report.missing_classes)
            verb = "never appears" if len(report.missing_classes) == 1 else "never appear"
            raise EngineError(
                f"the data declares n_class={self.spec.data.n_class}, but "
                f"{described} {verb} in the training masks - checked "
                f"{report.samples_checked} of {report.total_samples}, and "
                f"{report.unmatched:.1%} of their voxels matched no entry.\n"
                f"  A class with no examples has no gradient towards it: the run would "
                f"finish, the loss would fall, and the model would answer 'background' "
                f"everywhere.\n"
                + (
                    "  A channel nothing matched is a channel of zeros: it trains to "
                    "predict nothing and scores a Dice of 0 while every other number "
                    "looks ordinary. Check the `of` values against the label map, or the "
                    "pattern against the mask files.\n"
                    if self.spec.data.multilabel
                    else "  Either the colormap or labels do not describe these masks, or "
                    "n_class counts a class the data does not contain.\n"
                )
                + f"  Classes that do appear: {report.present_classes}.\n"
                f"  To look at more of the data than this check does:\n"
                f"    engine.dataset(spec.models[0], 'train').inspect_masks(limit=500)\n"
                f"  If the class is real but rarer than that, Engine(..., "
                f"check_masks=False) proceeds - knowing it cannot be learned from the "
                f"examples present."
            )

        # Nested regions have no background entry on purpose, so every background voxel is
        # unmatched and the fraction sits near 1 on any dataset where the thing being found
        # is small. Warning there would fire on every correct BraTS run and teach the reader
        # to ignore the one case where it means something.
        if report.unmatched > _UNMATCHED_WARNING and not self.spec.data.multilabel:
            warnings.warn(
                f"model '{model_spec.name}': {report.unmatched:.1%} of the training "
                f"masks' voxels match no colormap entry or label, and become background. "
                f"Compression around a mask's edge accounts for well under 1%; this is "
                f"more than that. Check the colormap against the mask files.",
                stacklevel=3,
            )

    def _describe_classes(self, indices: list[int]) -> str:
        """Name a class by what the specification said it was, not by its number alone."""
        data = self.spec.data
        parts = []
        for index in indices:
            if data.regions is not None and index < len(data.regions):
                region = data.regions[index]
                parts.append(f"region '{region.name}' (of {list(region.of)})")
            elif data.classes_from is not None and index < len(data.classes_from):
                parts.append(f"class {index} (files matching '{data.classes_from[index]}')")
            elif data.label_map and data.labels is not None and index < len(data.labels):
                parts.append(f"class {index} (label {data.labels[index]})")
            elif data.colormap is not None and index < len(data.colormap):
                parts.append(f"class {index} (colour {tuple(data.colormap[index])})")
            else:
                parts.append(f"class {index}")
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    #: What a sidecar may fill in when the specification did not say. Deliberately not
    #: `input_shape` or `channels`: the rank comes from `input_shape` and is needed while the
    #: specification is validated - long before any weights exist - and the data pipeline
    #: reads both before a network is built. Making them wait on a sidecar would mean a
    #: specification that cannot be checked without a download, which is the air-gapped
    #: hospital problem this project has carried since RECON.md.
    ADOPTABLE: ClassVar[dict[str, str]] = {
        "architecture": "architecture",
        "blocks": "blocks",
        "filters": "filters",
    }

    # -------------------------------------------------------------------- fit
    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        """Train every model the spec asks for, in order.

        Args:
            verbose: Print each model's name and its per-epoch numbers as they arrive.

        Returns:
            One `History` per model, by name. The same objects stay on the engine, so
            `evaluate`, `predict` and `export_weights` work afterwards.

        Raises:
            EngineError: If a split a model needs is missing, or carries no masks while
                something is about to score against it. `check_masks` is what refuses a
                declared class appearing in no mask: a target channel that is zero everywhere
                has no gradient towards it, so the model is never shown what to find.
        """
        for model_spec in self.spec.models:
            if model_spec.weights:
                model_spec, _ = self._adopt_from_weights(model_spec)
            if model_spec.fit:
                self._refuse_masks_that_describe_nothing(model_spec)
            if verbose:
                print(f"\n=== {model_spec.name} ({model_spec.architecture.value}) ===")

            network = build_model(
                model_spec, n_class=self.spec.data.n_class, encoder=_encoder_for(model_spec)
            )
            if model_spec.weights:
                self._load_weights(network, model_spec.weights, model_spec)

            trainer = Trainer(
                network,
                model_spec,
                device=self.device,
                multilabel=self.spec.data.multilabel,
                spacing=self.spec.data.target_spacing,
            )
            run = ModelRun(
                name=model_spec.name, spec=model_spec, model=trainer.model, trainer=trainer
            )

            if model_spec.fit:
                run.history = trainer.fit(
                    self.loader(
                        model_spec, "train", augmented=True, shuffle=self.spec.data.shuffle
                    ),
                    # None when the run says it has no validation set. `fit` already took
                    # an optional loader, so nothing below this had to learn about it - the
                    # history simply has no `val_` columns, which is the honest shape for a
                    # run that measured nothing.
                    self.loader(model_spec, "validation")
                    if "validation" in self._samples
                    else None,
                    verbose=verbose,
                    # The names the validation loader serves, in its order - the same list
                    # `evaluate_cases` builds, and safe for the same reason: that loader is
                    # never shuffled. With them the epoch can report its worst case, which
                    # is the number this package tells everybody to read and gave nobody a
                    # way to watch.
                    validation_cases=(
                        self._case_names(model_spec, "validation")
                        if "validation" in self._samples
                        else None
                    ),
                )
                run.trained = True
            self.runs[model_spec.name] = run
            self._record(run)

        return {name: run.history for name, run in self.runs.items()}

    def _record(self, run: ModelRun) -> None:
        """What this run leaves behind, when `output_dir` was asked for.

        Nothing is derived here that the specification does not already carry - the
        colormap, the input shape and the window are all in it - so the record is the
        specification plus the history. That is the honest shape for segmentation; the
        detection engine has anchors to add.
        """
        from pyplatypus.runs import wants_a_record, write_record

        if wants_a_record(self.spec):
            write_record(self.spec, run.name, history=run.history)

    # --------------------------------------------------------------- evaluate
    def _needs_masks(self, split: str) -> None:
        """What this engine calls a labelled split, and what to do instead."""
        self._needs_labels(
            split,
            f"predict('{split}') works on it, and save_masks() writes what it predicts.",
        )

    def evaluate(self, split: str = "validation", *, at: str = "model") -> list[dict[str, Any]]:
        """One row per model: the comparison table the whole multi-model idea is for.

        Args:
            split: Which data to score on.
            at: Which grid to score on. `"model"` is each model's own `load_shape` and the
                default, because changing it would silently restate every number anybody
                has already written down. `"source"` puts every prediction back on the grid
                of the file it came from and scores it against the truth **as stored**, so
                the rows answer one question.

        Returns:
            One row per model: what it is, how long it trained, its loss and its metrics.
            The `loss` column is comparable only between models trained on the same
            objective, which is why `loss_function` sits beside it - and the metrics are
            comparable only between models scored on the same grid, which is why
            `load_shape` sits beside them.

        `load_shape` is the second of those and the one nothing used to say. At
        `at="model"` each model is scored through its own loader, so a model at
        `input_shape: [512, 512]` is measured against a truth reduced to 512 while a model
        that tiles 2048 is measured against the truth as stored. Both numbers are a real
        Dice of a real prediction against a real mask; they are not answers to the same
        question. Measured on retinal vessels (`maju116/rplatypus#187`), that difference
        reported a resized model at **+0.0187 where the honest figure was +0.0033**, with
        the bias reversed on clDice because upsampling smooths a prediction - better
        against a skeleton, worse against a boundary. So when the rows do not share a grid
        this says so, once, rather than leaving a table that looks like a comparison.

        **`at="source"` is what that warning now points at.** The truth is never resampled -
        only predictions move - which is why the grid is the files' own rather than one the
        caller names: a named grid interpolates the very thing every model is being judged
        against. The column then reads `source`, because what it has always meant is the
        grid this row's numbers were computed on.

        Two things that are the price of it, neither recoverable: interpolating
        probabilities softens a boundary, and where the forward crop cut anatomy away the
        inverse pads it with background, which means *not examined* rather than *nothing
        there*. For a comparison the second is correct rather than unfair - a model whose
        field of view did not contain the organ did not segment it - but a 0.000 arriving
        that way is a result and not a bug.

        **The loss is not reported at `at="source"`.** A loss is defined by the objective a
        model was trained with, on the grid it was trained on; re-deriving one from a
        resampled prediction would be a number with no training meaning and a column
        somebody would compare.

        **The pooling is the same in both**: the mean over cases. That has been the claim
        here since `at` was added and became true in `0.8.0a40` - until then this branch
        passed no case names and reported the mean over *batches*, so the sentence described
        a pooling the code did not do and the comment in the `at="source"` branch rested on
        it. `maju116/rplatypus#196`.

        A case's own score is a ratio of sums - its tiles' counts added, the formula applied
        once - and the split's is the mean of those. `evaluate_cases` pools a **group** the
        same way, by adding the counts of the images in it, which is a different number again
        and the better statistic for a patient: `summarise_cases` is where to read it. So
        `at` changes the grid and nothing else, and the gap between its two values is the rig
        it exists to measure rather than that gap plus a change of definition.

        Raises:
            EngineError: If the split carries no masks - and the message says `predict`
                works on it, because that is the question asked next.
        """
        if not self.runs:
            raise EngineError("nothing has been trained or loaded yet; call fit() first")
        if at not in ("model", "source"):
            raise EngineError(f"at is 'model' or 'source', got '{at}'")
        self._needs_masks(split)

        grids = {run.name: run.spec.load_shape for run in self.runs.values()}
        if at == "model" and len(set(grids.values())) > 1:
            described = ", ".join(
                f"{name} at {' x '.join(str(size) for size in shape)}"
                for name, shape in grids.items()
            )
            warnings.warn(
                f"these models are scored on different grids ({described}), so their "
                "metrics are not comparable: each is measured against the truth reduced to "
                "its own `load_shape`. The `load_shape` column says which. To compare them, "
                "score them on one grid: `evaluate(at='source')` puts every prediction back "
                "on the grid of the file it came from and scores it against the truth as "
                "stored.",
                stacklevel=2,
            )

        table = []
        for run in self.runs.values():
            if at == "source":
                # The **mean over cases**, because that is what `at="model"` reports and
                # `at` must change exactly one thing. A ratio of sums over the split is the
                # better statistic and is what `evaluate_cases` pools a group with; the two
                # disagree here by 4e-5 on six similar cases and by more on unequal ones.
                # Making this one ratio-of-sums would mean the argument changed the grid
                # *and* the definition, so that the difference between its two values could
                # not be read as the rig it exists to measure. `maju116/rplatypus#196`.
                rows = self._score_at_source(run, split, lambda case: case)
                scores = {
                    name: sum(row[name] for row in rows) / len(rows)
                    for name in rows[0]
                    if name != "case"
                }
            else:
                # The case names, so this really is the mean over cases rather than the
                # mean over whatever the loader happened to batch. Until
                # `maju116/rplatypus#196` they were not passed, so this branch reported the
                # mean over **batches** - and the paragraph above, which says the pooling
                # is the same in both and has said so since `at` was added, was describing
                # a pooling the code did not do. The comment in the `at="source"` branch
                # rested on it too. A declaration beside a mechanism that does not match
                # it, in the one place the two values of one argument are meant to differ
                # by the grid alone.
                scores = run.trainer.evaluate(
                    self.loader(run.spec, split), self._case_names(run.spec, split)
                )
            best = run.history.best("val_loss")
            table.append(
                {
                    "model": run.name,
                    "architecture": run.spec.architecture.value,
                    # Named, because a loss column is only comparable between models that
                    # were trained on the same one - see best_model().
                    "loss_function": run.spec.loss.name,
                    "parameters": run.parameters,
                    # The grid this row's numbers were computed on. Beside the metrics for
                    # the same reason `loss_function` sits beside `loss`: it is what decides
                    # whether two rows may be compared, and a table that does not carry it
                    # cannot be read correctly a month later.
                    #
                    # A string rather than the tuple, because this table crosses into R as a
                    # data frame and a list column is not plain data - `test_the_table_is
                    # _plain_data_for_the_trip_into_r` is the test that says so, and it
                    # caught this the first time it ran.
                    "load_shape": (
                        "source"
                        if at == "source"
                        else " x ".join(str(size) for size in run.spec.load_shape)
                    ),
                    "epochs_run": len(run.history),
                    "best_epoch": best["epoch"] if best else None,
                    **{key.removeprefix("val_"): value for key, value in scores.items()},
                }
            )
        return table

    def evaluate_cases(
        self,
        model_name: str | None = None,
        split: str = "validation",
        *,
        group_by: str | None = None,
        at: str = "model",
    ) -> list[dict[str, Any]]:
        """One row per case instead of one row per model.

        The comparison table answers "which model"; this answers "on whom does it fail",
        which is the question a clinician asks first and the one a single mean cannot
        answer. With `group_by`, each row also carries the group - the patient, usually -
        so the rows can be summarised per patient rather than per slice.

        Args:
            model_name: Which model, by name. Defaults to the first.
            split: Which data to score on.
            group_by: A pattern picking a group out of each case's name. With it the rows
                are patients rather than slices: a patient's slices are pooled into one
                score, the way a volume would be.
            at: Which grid to score on, as for `evaluate`. `"source"` reads each truth as
                stored and scores the prediction there, which is the only form in which
                rows from two models - or from two runs with different `target_spacing` -
                mean the same thing.

        Returns:
            One row per case, or per group. `summarise_cases` turns them into the
            distribution - and its minimum is the number worth reading.
        """
        model_name = self._resolve_model(model_name)
        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        if not run.spec.metrics:
            raise EngineError(
                f"'{model_name}' has no metrics, so there is nothing to report per case. "
                "Add at least one, for example metrics: [{name: dice}]."
            )
        if at not in ("model", "source"):
            raise EngineError(f"at is 'model' or 'source', got '{at}'")
        self._needs_masks(split)

        if at == "source":
            key = "group" if group_by else "case"
            rows = self._score_at_source(
                run,
                split,
                (lambda case: group_of(case, group_by)) if group_by else (lambda case: case),
            )
            return [{key: row.pop("case"), **row} for row in rows]

        cases = self._case_names(run.spec, split, group_by=group_by)
        # Never shuffled: the case names line up with the order examples arrive in.
        loader = self.loader(run.spec, split, shuffle=False, for_loss=False)
        rows = run.trainer.score_cases(loader, cases)
        key = "group" if group_by else "case"
        return [{key: row.pop("case"), **row} for row in rows]

    def _case_names(self, model, split: str, *, group_by: str | None = None) -> list[str]:
        """One name per example the loader will serve, in its order.

        A tiled run serves several examples per sample, so a case's name repeats - which is
        what lets the counts be pooled back into one score per case. Safe only where the
        loader is not shuffled, which is every split but training.
        """
        dataset = self.dataset(model, split)
        tiles = dataset.tiles_per_sample
        return [
            group_of(sample.key, group_by) if group_by else sample.key
            for sample in dataset.samples
            for _ in range(tiles)
        ]

    def _score_at_source(self, run, split: str, bucket) -> list[dict[str, Any]]:
        """Metrics on the grid of the file each prediction came from, bucket by bucket.

        `bucket` maps a case name to the row it belongs in: itself for `evaluate_cases`,
        its group when `group_by` says so, and one name for everything when `evaluate`
        wants the split's figure. One function rather than three, because the pooling rule
        is the thing that must not exist in several versions.

        **Streamed, one case at a time.** The predictions are at full resolution and tiling
        exists for images too large to resize, so this is exactly where holding the split
        would hurt most - 200 FIVES retinas at 2048 are 6.7 GB of probabilities.

        **No tiles here.** A prediction arrives whole from `predict_stream`, so every
        metric is measured once per case and the tile pooling `score_cases` does has
        nothing to do. What is left is the pooling across cases, which is the same rule it
        uses: a counting metric adds TP, FP and FN and applies its formula once, and a
        whole-mask metric is the mean of its cases, which is a weaker statement and is
        reported as one rather than dressed up as a pooled figure.

        **The spacing is each file's own**, which is what makes `surface_dice` mean
        millimetres here. On the model's grid every case shares `target_spacing` by
        construction; on this grid they do not, so the metrics are built per case.
        """
        import torch

        from pyplatypus.objectives import build_metrics
        from pyplatypus.objectives import functional as f

        dataset = self.dataset(run.spec, split)
        samples = self._samples[split]
        multilabel = self.spec.data.multilabel
        rank = run.spec.rank

        totals: dict[str, list[torch.Tensor]] = {}
        whole: dict[str, dict[str, list[float]]] = {}
        order: list[str] = []

        stream = self.predict_stream(run.name, split, space="source")
        with torch.no_grad():
            for sample, (case, probabilities) in zip(samples, stream, strict=True):
                # The stream yields its own case names and this walks the sample list
                # beside it; if the two ever disagree every score after it is against the
                # wrong mask, and nothing downstream could tell.
                if case != sample.key:
                    raise EngineError(
                        f"the prediction stream gave '{case}' where the samples give "
                        f"'{sample.key}'; the two are out of step and a score would be "
                        "computed against another case's mask"
                    )
                target = dataset.target_as_stored(sample)
                if target.shape[:-1] != probabilities.shape[:-1]:
                    raise EngineError(
                        f"'{case}': the prediction is {probabilities.shape[:-1]} and the "
                        f"truth as stored is {target.shape[:-1]}. These are different "
                        "pictures and scoring them would compare nothing."
                    )

                prediction = torch.from_numpy(
                    np.ascontiguousarray(np.moveaxis(probabilities, -1, 0))
                )[None]
                truth = torch.from_numpy(np.ascontiguousarray(np.moveaxis(target, -1, 0)))[None].to(
                    prediction.dtype
                )
                hard = (
                    (prediction > 0.5).to(truth.dtype)
                    if multilabel
                    # Probabilities rather than logits - `predict_stream` has activated
                    # them - so the threshold is 0.5 and the exclusive case is an argmax.
                    else f.as_onehot(prediction.argmax(dim=1), prediction.shape[1])
                )
                metrics = build_metrics(
                    run.spec.metrics,
                    multilabel=multilabel,
                    spacing=self._source_spacing(sample) if rank == 3 else None,
                )
                tp, fp, fn = f.overlaps(hard, truth)

                name = bucket(case)
                if name not in totals:
                    totals[name] = [torch.zeros_like(tp[0]) for _ in range(3)]
                    order.append(name)
                for slot, value in enumerate((tp, fp, fn)):
                    totals[name][slot] += value[0]
                for metric_name, metric in metrics.items():
                    if metric.accumulates:
                        continue
                    whole.setdefault(metric_name, {}).setdefault(name, []).append(
                        float(metric.coefficient(hard, truth)[0])
                    )

        built = build_metrics(run.spec.metrics, multilabel=multilabel, spacing=None)
        rows = []
        for name in order:
            row: dict[str, Any] = {"case": name}
            for metric_name, metric in built.items():
                if metric.accumulates:
                    row[metric_name] = metric.reduce(metric.combine(*totals[name])).item()
                else:
                    scores = whole[metric_name][name]
                    row[metric_name] = sum(scores) / len(scores)
            rows.append(row)
        return rows

    # ---------------------------------------------------------------- predict
    def predict(self, model_name: str | None = None, split: str = "test", *, space: str = "model"):
        """Class probabilities, channels-last, one array per source image.

        Tiles are reassembled, so an image that went in at 2048x1536 comes back at
        2048x1536 rather than as 24 unrelated pieces.

        `space` decides which grid the answer is on, and the two are different enough that the
        return type differs with it:

        * `"model"` - one stacked array, every prediction on the model's grid. What training
          saw, and the only form that can be a single array, since a stack requires one shape.
        * `"source"` - a **list**, one array per sample, each on the grid of the file it came
          from. What a person asked for when they asked to segment their scan: it can be laid
          over that scan, written beside it, or measured in millilitres of its voxels. Sources
          differ in size, so this cannot be stacked, and pretending otherwise by silently
          resizing is how a mask ends up describing the wrong anatomy.

        Args:
            split: Which data to predict on.
            model_name: Which model, by name. Defaults to the first.
            space: `"model"` returns one stacked array on the network's own grid.
                `"source"` maps each prediction back onto the grid of the scan it came
                from, undoing the resampling and the crop, and returns a **list** - scans
                differ in size, a stacked array needs one shape, and quietly resizing them
                to match is how a mask ends up describing anatomy it was not computed from.

        Returns:
            An array for `space="model"`, a list for `space="source"`.

        Two losses in source space: interpolation softens a boundary, and where the forward
        crop cut anatomy away the inverse pads with background - which means *not examined*
        rather than *nothing there*.
        """
        run, loader = self._prediction_loader(model_name, split, space)
        predictions = run.trainer.predict(loader)

        if space == "model":
            return predictions
        return [
            self._to_source_space(predictions[index], sample, run.spec)
            for index, sample in enumerate(self._samples[split])
        ]

    def _prediction_loader(self, model_name: str | None, split: str, space: str):
        """The run and a loader that will not reorder the tiles. Shared by both predicts."""
        model_name = self._resolve_model(model_name)
        if space not in ("model", "source"):
            raise EngineError(f"space is 'model' or 'source', got '{space}'")

        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        only_images = split == "test" and "test" in self._samples
        # Never shuffle: stitching depends on tiles arriving in the order they were cut.
        loader = self.loader(
            run.spec, split, shuffle=False, only_images=only_images, for_loss=False
        )
        return run, loader

    def predict_stream(
        self, model_name: str | None = None, split: str = "test", *, space: str = "model"
    ) -> Iterator[tuple[str, np.ndarray]]:
        """`(case, prediction)` one source image at a time, instead of all of them at once.

        `predict` returns a stacked array, so the whole split has to fit in memory. For a
        tiled split that is the whole thing at full resolution - 200 FIVES retinas at 2048
        is 6.7 GB for the result alone - and tiling exists precisely for images too large
        to resize, so the stacked form is least usable exactly where `splits` is most
        needed. This yields one image, lets the caller keep what it wants, and drops it.

        The case name comes with the prediction rather than being left to the caller to
        line up against `samples`: predictions arrive in the dataset's order, and anything
        that pairs two sequences by position is one `drop_last` away from scoring the wrong
        image. `group_of` turns a case into a group the same way `evaluate_cases` does.

        Args:
            model_name: Which model, by name. Defaults to the first.
            split: Which data to predict on.
            space: As for `predict` - `"model"` for the network's grid, `"source"` for the
                grid of the file each prediction came from.

        Returns:
            An iterator of `(case, prediction)` in the dataset's order. An iterator rather
            than a list on purpose: a list would hold the whole split again.
        """
        # Resolved here rather than inside the generator: a generator function runs none of
        # its body until the first `next`, so a mistyped model name would be reported after
        # the caller had already opened the file it was going to write into.
        run, loader = self._prediction_loader(model_name, split, space)
        return self._stream_predictions(run, loader, split, space)

    def _stream_predictions(self, run, loader, split: str, space: str):
        samples = self._samples[split]
        count = 0
        for index, prediction in enumerate(run.trainer.predict_stream(loader)):
            if index >= len(samples):
                raise EngineError(
                    f"the loader produced more images than the {len(samples)} samples in "
                    f"'{split}', so the case names cannot be trusted"
                )
            sample = samples[index]
            if space == "source":
                prediction = self._to_source_space(prediction, sample, run.spec)
            count += 1
            yield sample.key, prediction
        if count != len(samples):
            raise EngineError(
                f"predicted {count} images for the {len(samples)} samples in '{split}'; "
                "the loader must not drop any, or the case names are wrong"
            )

    def _to_source_space(
        self, prediction: np.ndarray, sample: Sample, model: SegmentationModel
    ) -> np.ndarray:
        """Undo what reading did, so the answer lands on the grid the data arrived on.

        The forward path is: read at native size, resample to `target_spacing` if asked, then
        crop or pad to the model's shape. This walks back out of it - crop or pad to the shape
        resampling produced, then resize to the source's own shape - and the result is exactly
        the source's shape by construction rather than by rounding.

        Two things are lost on the way back and neither can be recovered, so both are worth
        knowing. Interpolating probabilities smooths them, so a boundary returns slightly
        softer than the model drew it. And where the forward crop cut anatomy away, the inverse
        pads it with background: that padding means *not examined*, not *nothing there*. Give
        the model an `input_shape` that covers the anatomy if that distinction matters.
        """
        from pyplatypus.data.images import resize_image
        from pyplatypus.data.volumes import crop_or_pad, resize_volume

        shape = self._source_shape(sample)

        if model.rank == 3 and self.spec.data.target_spacing is not None:
            from pyplatypus.data.volumes import resample_to_spacing

            spacing = self._source_spacing(sample)
            # The shape resampling produced on the way in, computed the same way, so the crop
            # below is the exact inverse of the pad that happened there (and vice versa).
            resampled = resample_to_spacing(
                np.zeros((*shape, 1), dtype=np.float32),
                spacing,
                self.spec.data.target_spacing,
            ).shape[:3]
            prediction = crop_or_pad(prediction, resampled)

        if tuple(prediction.shape[:-1]) == tuple(shape):
            return prediction
        if model.rank == 3:
            return resize_volume(prediction, shape)
        return resize_image(prediction, shape)

    def _source_shape(self, sample: Sample) -> tuple[int, ...]:
        """The native shape of a sample, whatever it is made of."""
        from pyplatypus.data.channels import match_channels
        from pyplatypus.data.dicom_series import series_shape
        from pyplatypus.data.images import spatial_shape

        if _is_series(sample.images):
            return series_shape(list(sample.images))
        if self.spec.data.channels_from is not None and len(sample.images) > 1:
            ordered = match_channels(sample.images, self.spec.data.channels_from, key=sample.key)
            return spatial_shape(ordered[0])
        return spatial_shape(sample.images[0])

    def _source_spacing(self, sample: Sample):
        from pyplatypus.data.channels import match_channels
        from pyplatypus.data.dicom_series import series_spacing
        from pyplatypus.data.volumes import volume_spacing

        if _is_series(sample.images):
            return series_spacing(list(sample.images))
        if self.spec.data.channels_from is not None and len(sample.images) > 1:
            ordered = match_channels(sample.images, self.spec.data.channels_from, key=sample.key)
            return volume_spacing(ordered[0])
        return volume_spacing(sample.images[0])

    def best_model(self, key: str = "dice", split: str = "validation") -> str:
        """Rank the models on one column.

        Refuses to rank on `loss` when the models were not trained on the same one. Two
        losses are two different scales: a Focal-Tversky of 0.05 is not better than a
        CCE-Dice of 0.14, it is not even the same question. Metrics are comparable
        because they measure the mask, not the objective.

        Args:
            key: Which column to rank on.
            split: Which data to rank on.

        Returns:
            The winning model's name.

        Raises:
            EngineError: If there is no such column; the message lists the ones there are.
        """
        table = self.evaluate(split)
        if key == "loss":
            used = {row["loss_function"] for row in table}
            if len(used) > 1:
                raise EngineError(
                    "cannot rank by loss: these models were trained on different losses "
                    f"({', '.join(sorted(used))}), which are not on a common scale. "
                    "Rank on a metric instead, for example best_model('dice')."
                )
        lower_is_better = key.endswith("loss")
        if key not in table[0]:
            raise EngineError(
                f"no column '{key}' in the evaluation table; available: "
                f"{', '.join(k for k in table[0] if isinstance(table[0][k], float))}"
            )
        chosen = (min if lower_is_better else max)(table, key=lambda row: row[key])
        return chosen["model"]


def _encoder_for(spec: SegmentationModel) -> PretrainedEncoder | None:
    """The encoder a spec asks for, or None to let the model build its own.

    The block options are passed through so the one stage we own matches the rest of the
    network; the backbone's own layers are whatever they were trained as.
    """
    if spec.encoder is None:
        return None
    return PretrainedEncoder(
        spec.encoder,
        in_channels=spec.channels,
        blocks=spec.blocks,
        filters=spec.filters,
        pretrained=spec.pretrained,
        rank=spec.rank,
        width=spec.block_width,
        batch_norm=spec.batch_normalization,
        separable=spec.separable_conv,
        act=spec.activation,
        drop=spec.dropout,
        spatial_dropout=spec.spatial_dropout,
    )


def summarise_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn per-case rows into the distribution, one row per metric.

    What a paper reports: not a single Dice, but its mean, its spread and its worst case.
    The minimum is the one nobody publishes and everybody should - a model averaging 0.86
    that scores 0.11 on some patient has a failure mode, and the mean is where it hides.

    Standard deviation is the sample one (n-1), and is None for a single case, because the
    spread of one number is not zero, it is unknown.

    Args:
        rows: One dict per case, as `Engine.evaluate_cases` returns - a `case` column
            naming it, or `group` when the rows were pooled by `group_by`, and a float
            column per metric. Columns that are not floats are passed over.

    Returns:
        One dict per metric, with `metric`, `n`, `mean`, `sd`, `median`, `min`, `max`
        and `worst_case` - or `worst_group` when the rows were grouped, because naming
        it `worst_case` there would describe the wrong thing.

    Raises:
        EngineError: If there are no rows, or no column is a float - "nothing to
            summarise" and "nothing here is a number" are different problems.

    >>> rows = [
    ...     {"case": "patient01", "dice": 0.91},
    ...     {"case": "patient02", "dice": 0.88},
    ...     {"case": "patient03", "dice": 0.11},
    ... ]
    >>> summary = summarise_cases(rows)[0]
    >>> summary["metric"], summary["n"]
    ('dice', 3)
    >>> round(summary["mean"], 3), summary["min"], summary["worst_case"]
    (0.633, 0.11, 'patient03')

    The mean is 0.63 and one case sits at 0.11, which is the argument for reporting the
    minimum: over a hundred cases the mean hides it completely.

    Grouped rows are named for what they are:

    >>> grouped = [{"group": "p01", "dice": 0.9}, {"group": "p02", "dice": 0.4}]
    >>> sorted(summarise_cases(grouped)[0])[-1]
    'worst_group'

    The spread of one number is unknown rather than zero:

    >>> summarise_cases([{"case": "only", "dice": 0.9}])[0]["sd"] is None
    True
    """
    if not rows:
        raise EngineError("there are no case scores to summarise")

    label = "group" if "group" in rows[0] else "case"
    metrics = [key for key, value in rows[0].items() if isinstance(value, float)]
    if not metrics:
        raise EngineError("these rows carry no metric columns")

    out = []
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=float)
        worst = min(rows, key=lambda row: row[metric])
        out.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                f"worst_{label}": worst[label],
            }
        )
    return out


def labels_in_test_split(data, *, columns: Sequence[str] | None = None) -> bool:
    """Whether a test split carries labels for anything, without reading any of them.

    The question is about the *layout* - is the label subdirectory there and not empty,
    does the CSV name the column - and the layout is identical whether those files are
    united into one mask or read as one object each. What differs is everything after they
    are read.

    `columns` is which CSV columns a label would be read from, defaulting to the one
    `label_column` names. A label that is a **set** has one column per class, so there are
    several, and the answer is whether the header names **any** of them: `_discover_test`
    is built on "none" and "some" being different things, and a file naming two of three
    classes carries labels for something. The labelled read then refuses it by naming the
    missing column, which is what should happen - the fallback is for a file that carries
    no labels at all.

    A function rather than a method on `EngineBase`, and the reason is testable rather
    than stylistic: it takes a **data block**, so the question can be asked of a
    specification with nothing built - which is how `#320`'s own test asks it, over header
    shapes an engine cannot be constructed from at all, because constructing one is what
    refuses.

    **All four engines call this, and the docstring said for several releases that
    detection "answers this differently and must not be changed on the way past".** That
    was false when it was written: detection's own copy was this body verbatim, modulo one
    `return False` folded into splitting an empty string, and the fields it reads -
    `label_column`, `subdirs[1]` - are derived per task, which is what makes one function
    serve all of them. Classification is the one that genuinely differs and it differs
    *before* this call: its directories are the classes, so in `nested_dirs` mode the
    split is labelled by construction and it answers without asking the layout anything.
    """
    from pyplatypus.spec.common import DataMode

    path = Path(data.test_path)
    if data.mode is DataMode.CONFIG_FILE:
        if not path.is_file():
            return False
        header = path.read_text().splitlines()[:1]
        if not header:
            return False
        named = [name.strip() for name in header[0].split(",")]
        return any(column in named for column in columns or [data.label_column])
    if not path.is_dir():
        return False
    wanted = data.subdirs[1]
    return any(
        (entry / wanted).is_dir() and any((entry / wanted).iterdir())
        for entry in path.iterdir()
        if entry.is_dir()
    )


def split_from_train(data, discover, *, strict: bool, stratify: bool = False) -> dict[str, tuple]:
    """Divide the training folder, when `split` says to instead of naming more paths.

    Only the split case: the three-paths case stays with each engine, because what a test
    split means differs between them - segmentation asks for masks, detection asks for
    annotations and tolerates their absence.

    Divided in memory. `split_dataset()` remains the tool when the CSVs are the point -
    something to keep, to hand to a colleague, to cite - but a specification should not
    leave files behind as a side effect of being read.

    `stratify` picks `stratified_split` instead, which gives every class to every part. It
    is a parameter rather than two call sites because what differs is one function: the
    classification engine passes True because a per-class metric refuses to average over a
    class a pass never held, and measured, a random 70/15/15 of 100 images at 3% prevalence
    leaves some part empty of the rare class 90% of the time.

    No check for an empty split here. `split_samples` guarantees at least one sample in
    every split it is asked for, and refuses when there are fewer groups than splits or
    when train or validation is given a zero share - measured, not assumed. A guard
    restating that would be a second place for one rule and could never fire.
    """
    from pyplatypus.data.splits import split_samples, stratified_split

    found = discover(data.train_path, data, strict=strict).samples
    divide = stratified_split if stratify else split_samples
    divided = divide(
        found, fractions=data.split.fractions, group_by=data.split.group_by, seed=data.split.seed
    )
    samples = {"train": divided.train, "validation": divided.validation}
    if divided.test:
        samples["test"] = divided.test
    return samples
