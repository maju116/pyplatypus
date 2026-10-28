"""One label per image - or a set of them - from a specification to a trained model.

This engine takes a `ClassificationSpec`, like its three siblings, and `build_engine(spec)`
returns it for `task: classification`. It took the two halves and a `_Pieces` stand-in until
the tag landed, which was deliberate: the tag goes last, after everything behind it works, so
that it never names a task the package cannot train.

What this engine does **not** do is weights, and that is said here rather than left to be
noticed. `export_weights`, `_load_weights` and the registry adoption on `EngineBase` are a
slice of their own. The reason is not that there are no published classification weights:
`pyplatypus#236` is exactly that explanation, written into the instance engine, true when
written and false one release later when the first file was published - so the absence is
recorded as work not done rather than as a conclusion about the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pyplatypus.data.augmentation import build_augmenter
from pyplatypus.data.classification import ClassBalance, ClassificationDataset, discover_labelled
from pyplatypus.engine import (
    EngineBase,
    EngineError,
    labels_in_test_split,
    split_from_train,
)
from pyplatypus.models.classifier import build_classifier
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.spec.spec import ClassificationSpec, PlatypusSpec
from pyplatypus.training.classification_trainer import ClassificationTrainer
from pyplatypus.training.torch_data import make_classification_loader
from pyplatypus.training.trainer import History, seed_everything


@dataclass
class ClassificationRun:
    """One model, trained or loaded, with what it produced."""

    spec: ClassificationModel
    model: Any
    trainer: ClassificationTrainer
    history: History = field(default_factory=History)

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class ClassificationEngine(EngineBase):
    """Fit, score and answer, for one label per image."""

    #: What a labelled split carries here, for the one message that names it.
    labels_are = "labels"

    def __init__(
        self,
        spec: PlatypusSpec,
        *,
        device: str | None = None,
        num_workers: int = 0,
        strict_data: bool = True,
    ):
        if not isinstance(spec, ClassificationSpec):
            # Named per task rather than as one sentence about "not classification", which
            # is the segmentation engine's reasoning: the other three are different
            # mechanisms and a message that lumped them together would send a reader to the
            # wrong engine.
            elsewhere = {
                "semantic_segmentation": "`Engine` trains that one.",
                "instance_segmentation": "`InstanceEngine` trains that one.",
                "object_detection": "`DetectionEngine` trains that one.",
            }
            raise EngineError(
                f"this engine trains classification; the specification's task is "
                f"'{spec.task.value}'. "
                + elsewhere.get(spec.task.value, "")
                + " `build_engine(spec)` picks by the tag, so holding a specification is "
                "enough."
            )
        # `models` is held to unique names by the specification itself, with `min_length=1`
        # beside it - so there is no guard here, and that is the point rather than an
        # omission: a second copy of a refusal is the thing §3F of the workbench's notes is
        # a list of. It was a guard while this engine took its pieces loose.
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.runs: dict[str, ClassificationRun] = {}
        seed_everything(spec.seed, deterministic=spec.deterministic)

        self._samples: dict[str, tuple]
        self._discover_splits(self._discover, strict=strict_data)

    # ------------------------------------------------------------------- discovery

    def _discover(self, path, data, *, only_images: bool = False, strict: bool = True):
        """`discover_labelled`, in the shape `EngineBase` hands a discovery function.

        `labels_from` is forwarded from the specification, which is the **only** place it
        is held: the reader takes the columns as an argument so that it can be exercised
        without a specification, and this is the one line that decides what a configured
        run reads. A second source for it is how the two would come to disagree.
        """
        return discover_labelled(
            path,
            data,
            only_images=only_images,
            strict=strict,
            label_columns=data.labels_from,
        )

    def _discover_splits(self, discover, *, strict: bool) -> None:
        """The base's own walk, except that a split cut out of the training folder is
        **stratified**.

        That is the one thing this engine must not inherit. Measured over 400 seeds, a
        random 70/15/15 of 100 images at 3% prevalence leaves some part with none of the
        rare class 91% of the time, 64% at 200 and 20% at 500 - and every per-class metric
        refuses to average over a class its pass never held, so the run dies at the end of
        the first epoch with the epoch already paid for.

        **`stratified_split` dispatches on the shape of the label and this line does not
        have to**, which is why there is no `multilabel` here: strata cannot serve a set -
        a film with three findings belongs to three of them - so a set of labels is placed
        one group at a time, rarest class first. Measured over 400 draws of four findings
        at 10%, 5%, 3% and 2%, a random split leaves some part holding one side only of
        some class 51.25% of the time at 500 images where that is 0.00%.
        """
        data = self.spec.data
        if data.split is None:
            super()._discover_splits(discover, strict=strict)
            return

        self.labelled = {"train"}
        if data.validation:
            self.labelled.add("validation")
        self._samples = split_from_train(data, discover, strict=strict, stratify=True)
        if "test" in self._samples:
            self.labelled.add("test")

    def _has_any_labels(self) -> bool:
        """Whether the test split carries labels for anything, without reading them.

        In `nested_dirs` mode the label *is* the directory, so a test split read that way
        is labelled by construction - this is the one task that answers without asking the
        layout anything, because its directories are the classes rather than a place the
        labels sit beside the images.

        A CSV may name no label column at all, and **that is the ordinary shape of a
        prediction set**: `key, images` and nothing else. This returned `True`
        unconditionally until `pyplatypus#320` while saying both halves of the distinction
        out loud, so `_discover_test`'s fallback was unreachable and the one task whose
        whole output is a label per image was the only one that could not predict on
        unlabelled images from a CSV. The reader could already do it - `only_images=True`
        on that same file accepts it - and nothing asked.

        **For a set of labels the columns are `labels_from`'s, one per class**, and the
        question is whether the header names any of them. A file naming two of three
        carries labels for something, so the labelled read refuses it by naming the third
        rather than quietly becoming an unlabelled split, which is `_discover_test`'s
        "none and some are not the same thing" seen from the side that decides.
        """
        from pyplatypus.spec.common import DataMode

        data = self.spec.data
        if data.mode is not DataMode.CONFIG_FILE:
            return True
        return labels_in_test_split(data, columns=data.labels_from)

    # -------------------------------------------------------------------- the data

    def dataset(self, model_name: str | None, split: str, *, augment: bool | None = None):
        """One split's samples, as a dataset for one model.

        `augment` defaults to augmenting the training split and nothing else, which is the
        rule every other engine here follows: a validation score measured on transformed
        images is a score for a dataset nobody has.
        """
        self._require_split(split)
        model_spec = self._model_spec(model_name)
        if augment is None:
            augment = split == "train"
        return ClassificationDataset(
            self._samples[split],
            model_spec,
            self.spec.data,
            augmenter=(
                build_augmenter(model_spec.augmentation, rank=model_spec.rank) if augment else None
            ),
            only_images=split not in self.labelled,
        )

    def loader(
        self,
        model_name: str | None,
        split: str,
        *,
        augment: bool | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Ordered and complete unless the caller says otherwise, which only `fit` does.

        **Both flags used to be derived from `split == "train"` here, and that was
        `pyplatypus#321`.** A method that asks what the model said got the loader built for
        optimising over, so `predict` came back in the shuffled order its own docstring
        promises the split's own order, and `answers` paired each key with **another
        image's** probabilities. Measured: two `predict(split="train")` calls on one split
        are a permutation of each other - the same multiset of rows, not one row in place -
        and at 16 samples with `batch_size: 8` nothing is dropped, so no count disagreed
        and nothing refused.

        The three other engines take `shuffle` from the caller and every asking site passes
        `False`; this one derived it. So this is not a new parameter but the shape its
        siblings already have, which is also what makes the defect legible: one question
        answered four times, three the same way, and the odd one out is the bug.
        `_has_any_labels` was the same shape one release earlier.

        **The two flags stay separate rather than travelling under one name**, and that is
        measured rather than tidy: `drop_last` was `training and len(base) > batch_size`,
        independent of `data.shuffle`, so keying it to shuffling would stop dropping the
        ragged last batch for anybody running `shuffle: false` - and a last batch of one is
        a batch-norm with nothing to normalise over, which is the whole reason the flag is
        here. This task's batches are images rather than tiles, so a dropped one costs a
        single example.

        The `len(base) > batch_size` guard stays inside rather than moving to the caller: a
        split smaller than one batch would otherwise be emptied by the flag `fit` always
        passes, and that is a property of the data rather than a decision `fit` is making.
        """
        model_spec = self._model_spec(model_name)
        base = self.dataset(model_name, split, augment=augment)
        return make_classification_loader(
            base,
            batch_size=model_spec.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last and len(base) > model_spec.batch_size,
        )

    def balance(self, split: str = "train") -> ClassBalance:
        """How many images each class has in a split, and how lopsided that is.

        Public, and on the engine rather than only on the dataset, because it is the number
        every decision about this task's metrics turned on - and the person who most needs
        it is the one who has just written a specification and not yet run anything.

        **An unlabelled split is refused rather than counted**, which became reachable with
        `pyplatypus#320`: a test set of images alone counted this way reports every class
        at zero, `never` lists all of them, and `labels_are_a_set` reads `False` whatever
        the specification says - three readings that look like findings about a dataset and
        are findings about there being no labels. §4 of the workbench's notes: a zero in a
        table is read as a result.

        The sentence is this method's own rather than `_needs_labels`'s, because that one
        says *there is nothing to score against* and is about asking a model for a number.
        This is a question about the data with no model in it, and the reading somebody
        needs here is the zero they would otherwise have believed.
        """
        self._require_split(split)
        if split not in self.labelled:
            raise EngineError(
                f"the '{split}' split has no labels, so there is no balance to report - "
                f"counted anyway it would say every class has zero images, which reads as "
                f"a finding about the dataset. This is the ordinary shape of a prediction "
                f"set; ask `balance` of a split that has labels, or add the label "
                f"column(s) to the file this split was read from."
            )
        return self.dataset(None, split, augment=False).balance()

    # ------------------------------------------------------------------------- fit

    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        """Every model in the run, in the order the specification names them."""
        histories: dict[str, History] = {}
        for model_spec in self.spec.models:
            network = build_classifier(model_spec, n_class=self.spec.data.n_class)
            trainer = ClassificationTrainer(
                network,
                model_spec,
                n_class=self.spec.data.n_class,
                multilabel=self.spec.data.multilabel,
                device=self.device,
            )
            validation = (
                self.loader(model_spec.name, "validation")
                if "validation" in self._samples
                else None
            )
            history = trainer.fit(
                # The one caller that is optimising rather than asking, so the one that
                # asks for shuffling and for a ragged last batch to be dropped.
                self.loader(
                    model_spec.name,
                    "train",
                    shuffle=self.spec.data.shuffle,
                    drop_last=True,
                ),
                validation,
                verbose=verbose,
            )
            self.runs[model_spec.name] = ClassificationRun(
                spec=model_spec, model=network, trainer=trainer, history=history
            )
            histories[model_spec.name] = history
        return histories

    # --------------------------------------------------------------------- results

    def evaluate(self, split: str = "validation") -> list[dict[str, Any]]:
        """One row per model: the loss and every metric it was asked for.

        A row rather than a number, and no `_worst` column, because a case here is one
        image - a split's worst case would be 0 for any model that gets one image wrong,
        which is the shape of a number that carries no information.

        **The metric columns carry bare names and the `split` column says which split**,
        which is the segmentation table's shape rather than a new one: a row reading
        `split: test` beside a column called `val_loss` is a row that has to be explained.
        The trainer prefixes everything `val_` because an epoch has two passes to tell
        apart; a table has a column for that instead.

        `loss_function` sits beside `loss` for the reason it does there: a loss column is
        only comparable between models trained on the same one. `architecture` is read
        plainly rather than through `.value` - for this task it is a string, `plain_cnn` or
        the backbone's name, and there is no enum to take a value off.
        """
        self._require_split(split)
        self._needs_labels(split, "predict")
        rows = []
        for name, run in self._trained().items():
            scored = run.trainer.evaluate(self.loader(name, split, augment=False))
            best = run.history.best("val_loss")
            rows.append(
                {
                    "model": name,
                    "architecture": run.spec.architecture,
                    "loss_function": run.spec.loss.name,
                    "parameters": run.parameters,
                    "split": split,
                    "epochs_run": len(run.history),
                    "best_epoch": best["epoch"] if best else None,
                    **{key.removeprefix("val_"): value for key, value in scored.items()},
                }
            )
        return rows

    def predict(self, model_name: str | None = None, split: str = "test") -> np.ndarray:
        """Probabilities, `(images, classes)`, in the split's own order."""
        name = self._resolve_model(model_name)
        self._require_split(split)
        return self.runs[name].trainer.predict(self.loader(name, split, augment=False))

    def predict_labels(
        self, model_name: str | None = None, split: str = "test"
    ) -> list[str] | list[list[str]]:
        """The class names, because a name is what a reader wanted.

        Read off `data.classes`, whose **order is the class index** - so this is the one
        place where the engine knows the right names and a caller passing their own could
        be wrong without anything saying so.

        **A list of names per image when the label is a set**, which is the trainer's own
        shape and is there for its reason: one name per image would have to pick which of
        the findings is *the* one, and the model was not asked that.
        """
        name = self._resolve_model(model_name)
        self._require_split(split)
        return self.runs[name].trainer.predict_labels(
            self.loader(name, split, augment=False), self.spec.data.classes
        )

    def answers(self, model_name: str | None = None, split: str = "test") -> list[dict[str, Any]]:
        """One row per image: its key, its answer, and how sure the model was.

        The key is what makes this joinable to anything else the caller holds, and it is
        the thing `rplatypus#223` was filed about on the segmentation side - a prediction
        returned as a bare sequence has to be paired by position.

        **A set of labels is a different row and not the same row with a longer string.**
        `labels` holds the findings the image actually carries and `predicted` the ones the
        model says it does, both as lists in `classes` order, and there is **no
        `probability` column at all**: that column is the chosen class's own number, and
        with independent questions nothing was chosen - the `p_<class>` columns are each
        class's answer and they are the whole of it. Inventing a single figure here would
        mean picking a finding to call the answer, which is the thing `predict_labels`
        refuses to do one layer down.
        """
        name = self._resolve_model(model_name)
        probabilities = self.predict(name, split)
        classes = self.spec.data.classes
        samples = self._samples[split]
        rows = []
        for sample, row in zip(samples, probabilities, strict=True):
            per_class = {
                f"p_{klass}": float(value) for klass, value in zip(classes, row, strict=True)
            }
            if self.spec.data.multilabel:
                rows.append(
                    {
                        "key": sample.key,
                        # In `classes` order rather than as they were written, so two rows
                        # are comparable, and a list rather than a joined string because a
                        # class name may contain anything a CSV header may.
                        # `None` rather than `[]` when nobody said, because an empty list
                        # here means *this image carries none of the findings* - a
                        # labelled image, and the commonest row in every dataset this
                        # reads. An unlabelled split arrives with `labels is None`, which
                        # is the reader's own marker for the distinction, and reporting it
                        # as `[]` would put a measured answer in a column nobody filled
                        # in. Reachable since `pyplatypus#320`; the single-label branch
                        # below has always said `None` here for the same reason.
                        "labels": (
                            None
                            if sample.labels is None
                            else [klass for klass in classes if klass in sample.labels]
                        ),
                        "predicted": [
                            klass
                            for klass, chance in zip(classes, row, strict=True)
                            if chance > 0.5
                        ],
                        **per_class,
                    }
                )
                continue
            chosen = int(row.argmax())
            rows.append(
                {
                    "key": sample.key,
                    "label": sample.label,
                    "predicted": classes[chosen],
                    "probability": float(row[chosen]),
                    **per_class,
                }
            )
        return rows

    def best_model(self, key: str = "auc", split: str = "validation") -> str:
        """The model with the highest score on `key`.

        `auc` rather than `accuracy` as the default, for the reason it is the default metric:
        measured at a prevalence of 3%, accuracy spans 0.0100 across models whose AUC spans
        0.4631 and is not even monotone in their discrimination, so choosing on it is
        choosing on noise with a model that has no signal in the lead.
        """
        rows = self.evaluate(split)
        column = key.removeprefix("val_")
        missing = [row["model"] for row in rows if column not in row]
        if missing:
            described = {
                "model",
                "architecture",
                "loss_function",
                "parameters",
                "split",
                "epochs_run",
                "best_epoch",
            }
            available = sorted(set(rows[0]) - described) if rows else []
            raise EngineError(
                f"no model reports '{column}'; {', '.join(missing)} would have to ask for it "
                f"under `metrics`. Available: {', '.join(available) or 'nothing'}"
            )
        return max(rows, key=lambda row: row[column])["model"]

    def write_history(self, out_dir: str | Path | None = None) -> dict[str, Path]:
        """One CSV per model, in `output_dir` unless told otherwise."""
        base = Path(out_dir if out_dir is not None else self.spec.output_dir)
        base.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, run in self._trained().items():
            path = base / f"{name}_history.csv"
            records = run.history.records
            if not records:
                continue
            columns = list(records[0])
            lines = [",".join(columns)]
            lines += [",".join(str(record.get(c, "")) for c in columns) for record in records]
            path.write_text("\n".join(lines) + "\n")
            written[name] = path
        return written

    # ------------------------------------------------------------------- internals

    def _model_spec(self, model_name: str | None) -> ClassificationModel:
        """The specification for a model, whether or not it has been trained.

        Separate from `_resolve_model`, which answers out of `self.runs`: a dataset can be
        asked for before anything is fitted, and that is how somebody looks at their data.
        """
        if model_name is None:
            return self.spec.models[0]
        for model_spec in self.spec.models:
            if model_spec.name == model_name:
                return model_spec
        named = ", ".join(model.name for model in self.spec.models)
        raise EngineError(f"no model named '{model_name}' in this run; it has {named}")

    def _trained(self) -> dict[str, ClassificationRun]:
        if not self.runs:
            raise EngineError("nothing has been trained yet; call `fit` first")
        return self.runs
