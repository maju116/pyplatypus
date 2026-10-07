"""Many models, one spec.

This is what `platypus_fit(spec)` calls. Each model gets its own data pipeline, because
each may want a different input size or a different tiling, and its own augmentation,
because that is part of the experiment. Validation never gets augmented - measuring a
model on distorted data measures the distortion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyplatypus.data.augmentation import build_augmenter
from pyplatypus.data.dataset import SegmentationDataset, _is_series
from pyplatypus.data.paths import Sample, discover
from pyplatypus.data.splits import group_of
from pyplatypus.errors import PlatypusError
from pyplatypus.models import build_model
from pyplatypus.models.encoders import PretrainedEncoder
from pyplatypus.objectives.losses import loss_needs_distance
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
            raise EngineError(
                f"this engine trains segmentation; the specification's task is "
                f"'{spec.task.value}'. The detection half of the spec exists and "
                f"validates, but nothing trains from it yet - until it does, a detector "
                f"is assembled by hand from pyplatypus.detection, the way "
                f"examples/detect_blood_cells.py does."
            )
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.check_masks = check_masks
        self.runs: dict[str, ModelRun] = {}
        seed_everything(spec.seed)

        self._samples: dict[str, tuple[Sample, ...]]
        self._discover_splits(discover, strict=strict_data)

    def _has_any_labels(self) -> bool:
        """Whether the test split carries masks for anything, without reading them."""
        from pyplatypus.spec.common import DataMode

        path = Path(self.spec.data.test_path)
        if self.spec.data.mode is DataMode.CONFIG_FILE:
            if not path.is_file():
                return False
            header = path.read_text().splitlines()[:1]
            if not header:
                return False
            return self.spec.data.label_column in [name.strip() for name in header[0].split(",")]
        if not path.is_dir():
            return False
        wanted = self.spec.data.subdirs[1]
        return any(
            (entry / wanted).is_dir() and any((entry / wanted).iterdir())
            for entry in path.iterdir()
            if entry.is_dir()
        )

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
                loss is `boundary`, each batch carries a third item - the signed distance
                map - computed in the loader's workers rather than in the loss, because
                the transform costs more than an epoch of a small 3D model and the paths
                that only score or predict should not pay for it.

        Returns:
            A `torch.utils.data.DataLoader`. Batches hold two items, or three when the
            loss asks for a distance map.
        """
        # `boundary` needs the signed distance map of its target and nothing else does, so
        # the loader computes it only when the loss asks - the transform costs more than an
        # epoch of a small 3D model and nobody else should pay for it. Read from the spec
        # rather than from the trainer's loss object, because the loader is built first.
        #
        # `for_loss=False` is for the paths that score or predict rather than optimise:
        # they never call the loss, so a map built for them would be pure waste - 700 ms a
        # sample at 128^3, on work that does not read it.
        return make_loader(
            self.dataset(model, split, augmented=augmented, only_images=only_images),
            batch_size=model.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            with_distance=for_loss and loss_needs_distance(model.loss),
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
        """What the data says the classes are. One of `colormap` and `labels` is set.

        `n_class` is here rather than on the model because the data decides it. It used to
        be a model field, which made it a setting that could disagree with its own data and
        needed a validator to say so; now there is nothing to disagree with. The count is
        still compared when weights are loaded - it is simply contributed by the half of the
        specification that knows it.
        """
        data = self.spec.data
        classes = {"n_class": data.n_class}
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
        """
        if not self.check_masks:
            return
        # Nearest-neighbour reading of a mask is cheap in 2D and not in 3D, where every
        # sample is a volume. Fewer samples there, enough that absence still means
        # something.
        limit = 8 if model_spec.rank == 3 else 50
        dataset = self.dataset(model_spec, "train")
        report = dataset.inspect_masks(limit=limit)

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
                f"  Either the colormap or labels do not describe these masks, or "
                f"n_class counts a class the data does not contain.\n"
                f"  Classes that do appear: {report.present_classes}.\n"
                f"  To look at more of the data than this check does:\n"
                f"    engine.dataset(spec.models[0], 'train').inspect_masks(limit=500)\n"
                f"  If the class is real but rarer than that, Engine(..., "
                f"check_masks=False) proceeds - knowing it cannot be learned from the "
                f"examples present."
            )

        if report.unmatched > _UNMATCHED_WARNING:
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
            if data.label_map and data.labels is not None and index < len(data.labels):
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
    ADOPTABLE = ("architecture", "blocks", "filters")

    def _adopt_from_weights(self, model_spec):
        """Fill the architecture the weights already describe, where the spec stayed silent.

        A published file knows its own geometry - `dsbowl-unet` records u_net, 4 blocks, 16
        filters - so requiring the caller to restate it means knowing the internals of
        somebody else's model in order to use it, and being refused for guessing wrong.

        `model_fields_set` is what makes this safe: it tells a value the user chose from a
        default that happened to be there. Anything stated is left alone and still has to
        agree, so this loosens what may be omitted and nothing about what is checked. The
        detection side has adopted anchors this way since 0.3.0a12; this is the same move
        applied to the rest of the fingerprint.
        """
        from pyplatypus.weights import describe, resolve_weights

        try:
            sidecar = describe(resolve_weights(model_spec.weights))
        except Exception:  # noqa: BLE001 - a bad reference is load_into's story to tell,
            return model_spec  # told with the message it has always given.
        if not sidecar:
            return model_spec

        adopted = {
            field: sidecar[field]
            for field in self.ADOPTABLE
            if field in sidecar and field not in model_spec.model_fields_set
        }
        if not adopted:
            return model_spec
        # Re-validated rather than copied, because the adopted values have to face the same
        # checks a written one would: `blocks` decides what input sizes are divisible.
        return type(model_spec).model_validate({**model_spec.model_dump(), **adopted})

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
                model_spec = self._adopt_from_weights(model_spec)
            if model_spec.fit:
                self._refuse_masks_that_describe_nothing(model_spec)
            if verbose:
                print(f"\n=== {model_spec.name} ({model_spec.architecture.value}) ===")

            network = build_model(
                model_spec, n_class=self.spec.data.n_class, encoder=_encoder_for(model_spec)
            )
            if model_spec.weights:
                self._load_weights(network, model_spec.weights, model_spec)

            trainer = Trainer(network, model_spec, device=self.device)
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

    def evaluate(self, split: str = "validation") -> list[dict[str, Any]]:
        """One row per model: the comparison table the whole multi-model idea is for.

        Args:
            split: Which data to score on.

        Returns:
            One row per model: what it is, how long it trained, its loss and its metrics.
            The `loss` column is comparable only between models trained on the same
            objective, which is why `loss_function` sits beside it.

        Raises:
            EngineError: If the split carries no masks - and the message says `predict`
            works on it, because that is the question asked next.
        """
        if not self.runs:
            raise EngineError("nothing has been trained or loaded yet; call fit() first")
        self._needs_masks(split)

        table = []
        for run in self.runs.values():
            scores = run.trainer.evaluate(self.loader(run.spec, split))
            best = run.history.best("val_loss")
            table.append(
                {
                    "model": run.name,
                    "architecture": run.spec.architecture.value,
                    # Named, because a loss column is only comparable between models that
                    # were trained on the same one - see best_model().
                    "loss_function": run.spec.loss.name,
                    "parameters": run.parameters,
                    "epochs_run": len(run.history),
                    "best_epoch": best["epoch"] if best else None,
                    **{key.removeprefix("val_"): value for key, value in scores.items()},
                }
            )
        return table

    def evaluate_cases(
        self, model_name: str, split: str = "validation", *, group_by: str | None = None
    ) -> list[dict[str, Any]]:
        """One row per case instead of one row per model.

        The comparison table answers "which model"; this answers "on whom does it fail",
        which is the question a clinician asks first and the one a single mean cannot
        answer. With `group_by`, each row also carries the group - the patient, usually -
        so the rows can be summarised per patient rather than per slice.

        Args:
            model: Which model, by name. Defaults to the first.
            split: Which data to score on.
            group_by: A pattern picking a group out of each case's name. With it the rows
                are patients rather than slices: a patient's slices are pooled into one
                score, the way a volume would be.

        Returns:
            One row per case, or per group. `summarise_cases` turns them into the
            distribution - and its minimum is the number worth reading.
        """
        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        if not run.spec.metrics:
            raise EngineError(
                f"'{model_name}' has no metrics, so there is nothing to report per case. "
                "Add at least one, for example metrics: [{name: dice}]."
            )
        self._needs_masks(split)

        dataset = self.dataset(run.spec, split)
        tiles = dataset.tiles_per_sample
        cases = [
            group_of(sample.key, group_by) if group_by else sample.key
            for sample in dataset.samples
            for _ in range(tiles)
        ]
        # Never shuffled: the case names line up with the order examples arrive in.
        loader = self.loader(run.spec, split, shuffle=False, for_loss=False)
        rows = run.trainer.score_cases(loader, cases)
        key = "group" if group_by else "case"
        return [{key: row.pop("case"), **row} for row in rows]

    # ---------------------------------------------------------------- predict
    def predict(self, model_name: str, split: str = "test", *, space: str = "model"):
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
            model: Which model, by name. Defaults to the first.
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
        predictions = run.trainer.predict(loader)

        if space == "model":
            return predictions
        samples = self._samples[split]
        return [
            self._to_source_space(predictions[index], sample, run.spec)
            for index, sample in enumerate(samples)
        ]

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


def split_from_train(data, discover, *, strict: bool) -> dict[str, tuple]:
    """Divide the training folder, when `split` says to instead of naming more paths.

    Only the split case: the three-paths case stays with each engine, because what a test
    split means differs between them - segmentation asks for masks, detection asks for
    annotations and tolerates their absence.

    Divided in memory. `split_dataset()` remains the tool when the CSVs are the point -
    something to keep, to hand to a colleague, to cite - but a specification should not
    leave files behind as a side effect of being read.

    No check for an empty split here. `split_samples` guarantees at least one sample in
    every split it is asked for, and refuses when there are fewer groups than splits or
    when train or validation is given a zero share - measured, not assumed. A guard
    restating that would be a second place for one rule and could never fire.
    """
    from pyplatypus.data.splits import split_samples

    found = discover(data.train_path, data, strict=strict).samples
    divided = split_samples(
        found, fractions=data.split.fractions, group_by=data.split.group_by, seed=data.split.seed
    )
    samples = {"train": divided.train, "validation": divided.validation}
    if divided.test:
        samples["test"] = divided.test
    return samples
