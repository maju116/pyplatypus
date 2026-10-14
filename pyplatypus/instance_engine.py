"""Instances, trained and decoded: the third engine.

The same shape as the other two - `fit`, `evaluate`, `evaluate_cases`, `predict`,
`export_weights`, the names doing the same jobs - over a head that is not the data's class
count and a score that is not an overlap.

**Three things here are genuinely different from semantic segmentation, and they are the
content of this module.**

The head is `f(separation, n_class)` rather than `n_class`, derived by the specification and
never stated. The objective is composite for `flows`: a regression term on the two vector
channels, masked to the foreground, plus an overlap term on the classes - which is why
`InstanceTrainer` exists and overrides exactly three things, the loss, the window the
metrics read, and the activation. And the score is computed **after** a post-process has
run, so it is not a function of `(logits, target)` and cannot be another entry in the
metrics registry; `evaluate` returns one row per (model, decoding) instead.

**What this engine is honestly worth is in `InstanceSpec`'s own description and is not
flattering.** On the Data Science Bowl the separation signal is worth +0.068 at best while a
perfect mask is worth +0.2398, so the mask is the limit by a factor of 3.6 and
`connected_components` - which ignores the signal entirely - is always in the table as the
row every other row is read against. Three signals were measured, two of them trained, and
all three agree. `JOURNAL.md` §4bq.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyplatypus.data.augmentation import build_augmenter
from pyplatypus.data.instances import InstanceDataset
from pyplatypus.data.paths import Sample, discover
from pyplatypus.data.splits import group_of
from pyplatypus.engine import EngineBase, EngineError, labels_in_test_split
from pyplatypus.instances.decode import decode_instances
from pyplatypus.instances.metrics import score_cases, summarise_instances
from pyplatypus.models import build_model
from pyplatypus.objectives import build_loss
from pyplatypus.objectives import functional as f
from pyplatypus.spec.instances import InstanceModel, Separation
from pyplatypus.spec.spec import InstanceSpec, PlatypusSpec
from pyplatypus.training.torch_data import make_loader
from pyplatypus.training.trainer import History, Trainer, seed_everything


class InstanceObjective(torch.nn.Module):
    """The composite objective a `flows` model is trained against.

    Two terms, summed: a regression on the vector channels and an overlap loss on the
    classes. They are separate because the channels are different kinds of thing - a
    direction is not a probability - and a single loss over all four would either activate
    the vectors or score the classes unactivated, each of which returns a number and a
    gradient and no error anywhere.

    `needs_mask` is how the loader is told to carry the foreground, and the foreground must
    come from the loader rather than from the target's own values: a flow field is exactly
    zero at every object's sink, so the non-zero voxels are the foreground minus precisely
    the seeds - and the seeds are what limits this family of methods (`pyplatypus#211`).

    The default weight of 1.0 is measured rather than conventional: over the 60-epoch run
    the two terms stay the same order of magnitude throughout, 0.5937 against 0.6345 after
    the first epoch and 0.0734 against 0.0659 at the last.
    """

    #: Always, because the regression term is the reason this class exists. Whether the
    #: term itself divides by the mask is `where`'s business, and `where: everywhere` still
    #: gets handed the mask and ignores it - one shape of batch for every flows run beats a
    #: loader whose batch width depends on a field two levels down.
    needs_mask = True

    def __init__(self, semantic, regression, *, weight: float, semantic_from: int):
        super().__init__()
        self.semantic = semantic
        self.regression = regression
        self.weight = float(weight)
        self.semantic_from = int(semantic_from)

    def forward(self, logits, target, mask=None):
        vectors = slice(0, self.semantic_from)
        classes = slice(self.semantic_from, None)
        flow = self.regression(logits[:, vectors], target[:, vectors], mask)
        semantic = self.semantic(logits[:, classes], target[:, classes])
        return self.weight * flow + semantic


class _MetricOnClasses(torch.nn.Module):
    """One metric, shown only the class channels.

    Every metric in this package is an overlap of a probability against a one-hot target,
    and a flow component is neither - so the metrics are given the class block and nothing
    else. Without this, foreground Dice on a `flows` model would be computed over a softmax
    taken across two directions and two classes, which is a number with no meaning that
    moves when the model improves.

    `accumulates` is proxied rather than defaulted: the trainer reads it to decide which
    metrics an epoch can honestly report on a tiled run, and a wrapper that answered for
    itself would quietly put `cldice` back into a tiled epoch.
    """

    def __init__(self, metric, start: int):
        super().__init__()
        self.metric = metric
        self.start = int(start)

    @property
    def accumulates(self) -> bool:
        return self.metric.accumulates

    def forward(self, logits, target):
        return self.metric(logits[:, self.start :], target[:, self.start :])


class InstanceTrainer(Trainer):
    """A `Trainer` that knows its head is two kinds of channel at once.

    Three overrides and nothing else: the objective, the window the metrics read, and the
    activation. Everything a training run needs beyond that - the epoch loop, the
    callbacks, the optimiser and its parameter groups, the freezing, the history, the
    stitching of tiles - is the same work and is inherited rather than written again.
    """

    def __init__(self, model, spec: InstanceModel, *, device=None, callbacks=None):
        super().__init__(model, spec, device=device, callbacks=callbacks, multilabel=False)
        self.separation = spec.signal
        start = spec.separation.semantic_from
        if spec.signal is Separation.FLOWS:
            self.loss_fn = InstanceObjective(
                build_loss(spec.loss, multilabel=False),
                build_loss(spec.objective, multilabel=False),
                weight=spec.separation.weight,
                semantic_from=start,
            )
            self.metrics = {
                name: _MetricOnClasses(metric, start) for name, metric in self.metrics.items()
            }
        # Recomputed because `loss_fn` was just replaced: the base class read the flag off
        # the semantic loss, which asks for no extra tensor.
        self._wants_extra = getattr(self.loss_fn, "needs_distance", False) or getattr(
            self.loss_fn, "needs_mask", False
        )

    def activate(self, out: torch.Tensor) -> torch.Tensor:
        """Classes to probabilities, vectors left exactly as the network produced them.

        A softmax across the whole head would make the two flow components sum towards one
        with the classes - a number, a gradient and no error anywhere. `rim`'s head is all
        classes, so there it is the inherited answer unchanged.
        """
        start = self.separation.semantic_from
        if start == 0:
            return f.probabilities(out, multilabel=False)
        return torch.cat([out[:, :start], f.probabilities(out[:, start:], multilabel=False)], dim=1)


@dataclass
class InstanceRun:
    """Everything one instance model left behind."""

    name: str
    spec: InstanceModel
    model: torch.nn.Module
    trainer: InstanceTrainer
    history: History = field(default_factory=History)
    trained: bool = False

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class InstanceEngine(EngineBase):
    """Many instance models, one spec. What `platypus_fit(spec)` will call for this task."""

    #: What a labelled split carries here, for the one message that names it.
    labels_are = "per-object masks"

    def __init__(
        self,
        spec: PlatypusSpec,
        *,
        device: str | None = None,
        num_workers: int = 0,
        strict_data: bool = True,
        check_objects: bool = True,
    ):
        if not isinstance(spec, InstanceSpec):
            raise EngineError(
                f"this engine trains instance segmentation; the specification's task is "
                f"'{spec.task.value}'. `semantic_segmentation` is `Engine` and "
                f"`object_detection` is `DetectionEngine`; `build_engine(spec)` picks by "
                f"the tag so that holding a specification is enough."
            )
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.check_objects = check_objects
        self.runs: dict[str, InstanceRun] = {}
        seed_everything(spec.seed, deterministic=spec.deterministic)

        self._samples: dict[str, tuple[Sample, ...]]
        self._discover_splits(discover, strict=strict_data)

    def _has_any_labels(self) -> bool:
        """Whether the test split carries masks for anything, without reading them.

        The same question and the same answer as the semantic engine's, from one place: a
        case's label subdirectory being there and not empty, or the CSV naming the column.
        Per-object files and a united mask are found identically - the difference is what
        happens after they are read - so answering it twice is how `EngineBase`'s two
        divergences happened.
        """
        return labels_in_test_split(self.spec.data)

    # ------------------------------------------------------------------- data
    def dataset(
        self,
        model: InstanceModel,
        split: str,
        *,
        augmented: bool = False,
        only_images: bool | None = None,
    ) -> InstanceDataset:
        """One split as (image, instance target) pairs.

        Args:
            model: Which model's settings to read - the input size, the tiling, the
                separation the target is built for and that target's own parameters.
            split: Which data.
            augmented: Apply the model's augmentation. Training only.
            only_images: Read images alone. Defaults to what the split actually has.

        Returns:
            An `InstanceDataset`. `inspect_objects()` on it is what `fit` calls to refuse a
            split whose objects have all vanished in the resize.
        """
        self._require_split(split)
        if only_images is None:
            only_images = split not in self.labelled
        augmenter = (
            build_augmenter(model.augmentation, model.rank, tuple(model.input_shape))
            if augmented
            else None
        )
        return InstanceDataset(
            self._samples[split],
            model,
            self.spec.data,
            augmenter=augmenter,
            only_images=only_images,
        )

    def loader(
        self,
        model: InstanceModel,
        split: str,
        *,
        augmented: bool = False,
        shuffle: bool = False,
        only_images: bool = False,
        for_loss: bool = True,
    ):
        """A torch DataLoader over one split, wrapping `dataset`.

        A `flows` run's batches carry a third item, the foreground the regression term is
        averaged over, and the paths that only score or predict do not pay for it -
        `for_loss=False` is how they say so. `rim` never needs one: its channels are all
        classes and its loss is an ordinary overlap.
        """
        return make_loader(
            self.dataset(model, split, augmented=augmented, only_images=only_images),
            batch_size=model.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            with_foreground=for_loss and model.signal is Separation.FLOWS,
        )

    # -------------------------------------------------------------------- fit
    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        """Train every model the spec asks for, in order.

        Args:
            verbose: Print each model's name and its per-epoch numbers as they arrive.

        Returns:
            One `History` per model, by name. The same objects stay on the engine, so
            `evaluate`, `predict` and `export_weights` work afterwards.

        Raises:
            EngineError: If a split a model needs is missing, carries no per-object masks,
                or carries objects that have all vanished at this input size - which
                `check_objects` is what looks for.
        """
        for model_spec in self.spec.models:
            if model_spec.fit and self.check_objects:
                self._refuse_a_split_that_lost_its_objects(model_spec)
            if verbose:
                print(
                    f"\n=== {model_spec.name} ({model_spec.architecture.value}, "
                    f"{model_spec.signal.value}) ==="
                )

            network = build_model(
                model_spec,
                n_class=model_spec.head_channels(self.spec.data.n_class),
                encoder=_encoder_for(model_spec),
            )
            if model_spec.weights:
                self._load_weights(network, model_spec.weights, model_spec)

            trainer = InstanceTrainer(network, model_spec, device=self.device)
            run = InstanceRun(
                name=model_spec.name, spec=model_spec, model=trainer.model, trainer=trainer
            )

            if model_spec.fit:
                run.history = trainer.fit(
                    self.loader(
                        model_spec, "train", augmented=True, shuffle=self.spec.data.shuffle
                    ),
                    self.loader(model_spec, "validation")
                    if "validation" in self._samples
                    else None,
                    verbose=verbose,
                )
                run.trained = True
            self.runs[model_spec.name] = run
            self._record(run)

        return {name: run.history for name, run in self.runs.items()}

    def _record(self, run: InstanceRun) -> None:
        from pyplatypus.runs import wants_a_record, write_record

        if wants_a_record(self.spec):
            write_record(self.spec, run.name, history=run.history)

    # ----------------------------------------------------------------- checks
    def _refuse_a_split_that_lost_its_objects(self, model_spec: InstanceModel) -> None:
        """Look at the objects before training on them.

        **The failure this prevents is silent and specific to this task.** An instance mask
        is one file per object and the objects are small - a Data Science Bowl nucleus runs
        from four pixels up - so an `input_shape` well below the source resolution can
        resize a case's objects out of existence. The target is then all background, the
        loss falls, every metric stays finite, and the run learns to answer "nothing here".

        Counted after the resize rather than on disk, because before it there is nothing
        wrong. A sample of cases, for the reason the mask check reads a sample: twenty cases
        answer the question and 670 of 375 files each is a minute of reading.
        """
        report = self.dataset(model_spec, "train").inspect_objects()
        if report.objects == 0:
            raise EngineError(
                f"'{model_spec.name}': none of the {report.cases_checked} training cases "
                f"checked (of {report.total_cases}) has a single object left at "
                f"input_shape={tuple(model_spec.input_shape)}.\n"
                f"  The objects are there on disk and the resize removed them: a label map "
                f"is read nearest-neighbour, so an object smaller than one output pixel "
                f"lands on no pixel at all.\n"
                f"  Train at a larger `input_shape`, or tile instead of resizing with "
                f"`splits`, which reads the source at splits x input_shape."
            )
        if report.empty:
            import warnings

            warnings.warn(
                f"'{model_spec.name}': {report.empty} of {report.cases_checked} training "
                f"cases checked have no object left at "
                f"input_shape={tuple(model_spec.input_shape)}; those cases train the model "
                f"to find nothing. The others hold {report.objects} objects, from "
                f"{report.smallest} to {report.largest} per case.",
                stacklevel=2,
            )

    def _needs_masks(self, split: str) -> None:
        self._needs_labels(
            split,
            f"predict('{split}') works on it, and predict_instances() decodes what it "
            f"predicts into objects.",
        )

    # ------------------------------------------------------------------ weights
    def _class_fingerprint(self) -> dict:
        """What the data says the classes are, which here is almost nothing - and the task.

        An instance dataset names no classes: the per-object files carry identity and no
        kind, so there is no colormap, no label list and no region vocabulary to record.
        `task` goes in instead, and it is the field that matters: without it a sidecar
        saying `n_class: 2` from this task and one from a two-class semantic run are
        indistinguishable, and a `flows` head only refuses to load into a semantic model
        because the channel counts happen to differ. The separation itself is on the
        model's own fingerprint.

        The asymmetry worth stating: the semantic engine does not write `task`, and a field
        absent from a sidecar is skipped rather than failed, so an older semantic weights
        file loads into a semantic model exactly as before.
        """
        return {"n_class": self.spec.data.n_class, "task": self.spec.task.value}

    def _load_weights(self, model, reference: str, spec: InstanceModel):
        """A registry name, a Hub reference, or a local path - see `pyplatypus.weights`.

        **Nothing is adopted from the file**, which is the one place this engine offers less
        than the semantic one. There are no published instance weights, so there is no
        registry entry to read `blocks` and `filters` out of; a local file still loads, and
        its sidecar still has to agree with the model it is being loaded into - including
        the separation, because a `rim` head and a `flows` head mean entirely different
        things by the same numbers.
        """
        from pyplatypus.weights import load_into

        return load_into(model, reference, spec, extra=self._class_fingerprint())

    def export_weights(self, model_name: str, path: str | Path, **extra) -> Path:
        """Write one trained model's weights, with the sidecar that says what they are for.

        Args:
            model_name: Which model's weights to write.
            path: Where to write. The sidecar goes beside it.
            **extra: Anything else to record - the dataset and its licence belong here.

        Returns:
            The path written.
        """
        from pyplatypus.weights import export_weights

        run = self._run(model_name)
        return export_weights(
            run.model, run.spec, path, extra={**self._class_fingerprint(), **(extra or {})}
        )

    # --------------------------------------------------------------- evaluate
    def _run(self, model_name: str | None) -> InstanceRun:
        model_name = self._resolve_model(model_name)
        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        return run

    def _decoded(self, run: InstanceRun, split: str):
        """Every case of a split, decoded every way the model asks, against its truth.

        One forward pass for all of them, which is the whole argument for `decode` being a
        list: the predictions are decoded as they arrive rather than held, so peak memory
        is one image's channels and one label map per decoding.

        Tiles are stitched by `predict_stream` before anything is decoded. That ordering is
        not luck - an object crossing a tile seam becomes two objects if the post-process
        runs per tile - and it is the reason this method exists rather than the decoding
        happening in the dataset.
        """
        dataset = self.dataset(run.spec, split)
        loader = self.loader(run.spec, split, shuffle=False, for_loss=False)
        blocks = run.spec.decodings
        rows: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
            block.name: [] for block in blocks
        }
        for index, prediction in enumerate(run.trainer.predict_stream(loader)):
            truth = dataset.labels_of(index)
            key = dataset.samples[index].key
            for block in blocks:
                predicted = decode_instances(prediction, run.spec.signal, block)
                rows[block.name].append((key, truth, predicted))
        return rows

    def evaluate(self, split: str = "validation") -> list[dict[str, Any]]:
        """One row per model **and decoding**: the comparison this task is for.

        A decoding costs a hundredth of an epoch, so the table gains an axis for nearly
        nothing.

        **This paragraph used to say `connected_components` is in every model's rows
        whether it was asked for or not, and it is not.** It is in the pair `decode` derives
        to when a model stayed silent, which is the signal's own decoding plus the
        yardstick; a model that *writes* a `decode` list gets the list it wrote and nothing
        added. That is the right behaviour - the engine does not put a row in somebody's
        table that they did not ask for - and the claim was wrong rather than the code: the
        only test on it covered the derived case, and
        `examples/compare_instance_separations.py` keeps the yardstick in its own `--decode`
        handling precisely because nothing below does it. `test_a_written_decode_list_is_what_you_get`
        is the half that was missing.

        The reason the yardstick belongs in a comparison is unchanged and is why the
        derivation includes it: it ignores the separation signal, so it is the row the
        signal's row has to be read against.

        Args:
            split: Which data to score on.

        Returns:
            One row per (model, decoding): what the model is, what it was trained on, the
            mask's own numbers, and the instance average precision with the object counts
            that explain it. `average_precision` is the mean over images, which is what the
            Data Science Bowl ranked on.

        Raises:
            EngineError: If nothing has been trained, or the split carries no masks.
        """
        if not self.runs:
            raise EngineError("nothing has been trained or loaded yet; call fit() first")
        self._needs_masks(split)

        table = []
        for run in self.runs.values():
            # The mask's own numbers, which are the same for every decoding of one model -
            # and the first thing to read: if the foreground moved, the instance number
            # moved for a reason that is not separation and no decoding can say so.
            scores = run.trainer.evaluate(self.loader(run.spec, split))
            best = run.history.best("val_loss")
            for name, cases in self._decoded(run, split).items():
                summary = summarise_instances(score_cases(cases))
                table.append(
                    {
                        "model": run.name,
                        "architecture": run.spec.architecture.value,
                        "separation": run.spec.signal.value,
                        "decode": name,
                        "loss_function": run.spec.loss.name,
                        "parameters": run.parameters,
                        "epochs_run": len(run.history),
                        "best_epoch": best["epoch"] if best else None,
                        "average_precision": summary["mean_average_precision"],
                        "truth_objects": summary["truth_objects"],
                        "predicted_objects": summary["predicted_objects"],
                        "worst_case": summary["worst"],
                        **{key.removeprefix("val_"): value for key, value in scores.items()},
                    }
                )
        return table

    def evaluate_cases(
        self,
        model_name: str | None = None,
        split: str = "validation",
        *,
        decode: str | None = None,
        group_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """One row per case instead of one row per model, for one decoding.

        One decoding rather than all of them, so a row stays one case and
        `summarise_cases` keeps meaning what it means. The table across decodings is
        `evaluate`'s job; this one answers "on which case does it fail", and the object
        counts beside each score are what say whether a case was mislabelled or shattered.

        Args:
            model_name: Which model, by name. Defaults to the first.
            split: Which data to score on.
            decode: Which decoding, by name. Defaults to the model's first, which is the
                separation's own rather than the yardstick.
            group_by: A pattern picking a group out of each case's name, pooling a group's
                objects into one score the way a volume's slices would be.

        Returns:
            One row per case, or per group.
        """
        run = self._run(model_name)
        self._needs_masks(split)
        names = [block.name for block in run.spec.decodings]
        wanted = names[0] if decode is None else decode
        if wanted not in names:
            raise EngineError(
                f"'{run.name}' does not decode by '{wanted}'; it asks for "
                f"{', '.join(names)}. A decoding has to be in the model's own `decode` "
                f"list, because the score depends on it and a row that silently used "
                f"another one would not say so."
            )

        cases = self._decoded(run, split)[wanted]
        if group_by:
            cases = _pooled_by_group(cases, group_by)
        rows = score_cases(cases)
        key = "group" if group_by else "case"
        return [{key: row.pop("case"), "decode": wanted, **row} for row in rows]

    def best_model(self, key: str = "average_precision", split: str = "validation") -> str:
        """Which model scored best, by one column of `evaluate`.

        The name alone, because a (model, decoding) pair is not a model: the rows are
        searched over both axes and the winning row's model is returned, so a model that
        wins only through `connected_components` is reported as the winner it is - which is
        the result this dataset actually produced.
        """
        table = self.evaluate(split)
        if key not in table[0]:
            raise EngineError(
                f"no column '{key}' in the table; it has: {', '.join(sorted(table[0]))}"
            )
        lower_is_better = key.endswith("loss")
        chosen = (min if lower_is_better else max)(table, key=lambda row: row[key])
        return chosen["model"]

    # ---------------------------------------------------------------- predict
    def predict(self, model_name: str | None = None, split: str = "test") -> np.ndarray:
        """The model's own channels, channels-last, one array per source image, stacked.

        The class channels are probabilities and the vector channels are **raw**, which is
        the one place this differs from the semantic engine's `predict`: a softmax across a
        flow component and a class would return a number and mean nothing.

        `predict_instances` is the form most callers want - this is what it decodes.
        """
        run = self._run(model_name)
        self._require_split(split)
        return run.trainer.predict(self.loader(run.spec, split, shuffle=False, for_loss=False))

    def predict_instances(
        self, model_name: str | None = None, split: str = "test", *, decode: str | None = None
    ) -> list[np.ndarray]:
        """One label map per source image: objects numbered from 1.

        A list rather than a stacked array even when every image is the same size, because
        what a caller does with these is read one case at a time - count the objects,
        measure them, draw them with `overlay_instances`.

        Args:
            model_name: Which model, by name. Defaults to the first.
            split: Which data to predict on.
            decode: Which decoding, by name. Defaults to the model's first.

        Returns:
            One `(H, W)` int32 label map per sample, in the split's own order.
        """
        run = self._run(model_name)
        self._require_split(split)
        names = [block.name for block in run.spec.decodings]
        wanted = names[0] if decode is None else decode
        blocks = [block for block in run.spec.decodings if block.name == wanted]
        if not blocks:
            raise EngineError(
                f"'{run.name}' does not decode by '{wanted}'; it asks for {', '.join(names)}"
            )

        loader = self.loader(run.spec, split, shuffle=False, only_images=True, for_loss=False)
        return [
            decode_instances(prediction, run.spec.signal, blocks[0])
            for prediction in run.trainer.predict_stream(loader)
        ]


def _pooled_by_group(cases, group_by: str):
    """Several cases' label maps as one pair, so a group scores once.

    The groups' objects are renumbered as they are pooled, which is the whole of the work:
    two slices of one patient both number their objects from 1, and stacking them without
    renumbering would merge object 1 of each into one object spanning both slices.
    """
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, truth, predicted in cases:
        pooled.setdefault(group_of(name, group_by), []).append((truth, predicted))
    rows = []
    for group, pairs in pooled.items():
        truths, predictions = [], []
        truth_offset = predicted_offset = 0
        for truth, predicted in pairs:
            truths.append(np.where(truth > 0, truth + truth_offset, 0))
            predictions.append(np.where(predicted > 0, predicted + predicted_offset, 0))
            truth_offset += int(truth.max())
            predicted_offset += int(predicted.max())
        rows.append((group, np.stack(truths), np.stack(predictions)))
    return rows


def _encoder_for(spec: InstanceModel):
    """The pretrained encoder a model asked for, if it asked for one.

    The same question the semantic engine asks and the same answer, reached through its own
    function rather than copied: an instance model is a U-shaped network with a wider head,
    so a timm backbone means here exactly what it means there.
    """
    from pyplatypus.engine import _encoder_for as semantic_encoder_for

    return semantic_encoder_for(spec)
