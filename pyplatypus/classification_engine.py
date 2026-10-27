"""One label per image, from a specification to a trained model and an answer.

**There is no top-level specification for this task yet**, which is why this engine takes
the two halves rather than a `ClassificationSpec`: `task: classification` lands last, after
everything behind it works, so that the tag never names a task the package cannot train.
Detection has been in the same state and says so in its own engine's refusal - *a detector
is assembled by hand from `pyplatypus.detection`* - so this is a state the package has had
before rather than an omission. `_Pieces` is the stand-in and goes away in that slice, when
`self.spec` becomes the real thing and `build_engine` learns the tag.

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
from pyplatypus.engine import EngineBase, EngineError, split_from_train
from pyplatypus.models.classifier import build_classifier
from pyplatypus.spec.classification import ClassificationData, ClassificationModel
from pyplatypus.training.classification_trainer import ClassificationTrainer
from pyplatypus.training.torch_data import make_classification_loader
from pyplatypus.training.trainer import History, seed_everything


@dataclass(frozen=True)
class _Pieces:
    """What a top-level specification would carry, until this task has one.

    `EngineBase` reads `self.spec.data` and nothing else, which is what makes standing in
    for a spec this cheap - measured by reading it rather than assumed.
    """

    data: ClassificationData
    models: list[ClassificationModel]
    output_dir: str = "."
    seed: int | None = None
    deterministic: bool = False


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
        data: ClassificationData,
        models: list[ClassificationModel],
        *,
        output_dir: str = ".",
        seed: int | None = None,
        deterministic: bool = False,
        device: str | None = None,
        num_workers: int = 0,
        strict_data: bool = True,
    ):
        if not models:
            raise EngineError("a run needs at least one model")
        names = [model.name for model in models]
        if len(set(names)) != len(names):
            raise EngineError(
                f"model names must be unique within a run and these are {', '.join(names)}; "
                "the name is what every result is keyed by"
            )
        self.spec = _Pieces(
            data=data,
            models=list(models),
            output_dir=output_dir,
            seed=seed,
            deterministic=deterministic,
        )
        self.device = device
        self.num_workers = num_workers
        self.runs: dict[str, ClassificationRun] = {}
        seed_everything(seed, deterministic=deterministic)

        self._samples: dict[str, tuple]
        self._discover_splits(self._discover, strict=strict_data)

    # ------------------------------------------------------------------- discovery

    def _discover(self, path, data, *, only_images: bool = False, strict: bool = True):
        """`discover_labelled`, in the shape `EngineBase` hands a discovery function."""
        return discover_labelled(path, data, only_images=only_images, strict=strict)

    def _discover_splits(self, discover, *, strict: bool) -> None:
        """The base's own walk, except that a split cut out of the training folder is
        **stratified**.

        That is the one thing this engine must not inherit. Measured over 400 seeds, a
        random 70/15/15 of 100 images at 3% prevalence leaves some part with none of the
        rare class 91% of the time, 64% at 200 and 20% at 500 - and every per-class metric
        refuses to average over a class its pass never held, so the run dies at the end of
        the first epoch with the epoch already paid for.
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
        """In `nested_dirs` mode the label *is* the directory, so a test split read that way
        is labelled by construction; a CSV may name no label column at all."""
        return True

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

    def loader(self, model_name: str | None, split: str, *, augment: bool | None = None):
        """Shuffled on the training split and ordered everywhere else.

        `drop_last` on training, and that is not a default copied from anywhere: a last
        batch of one is a batch-norm with nothing to normalise over, and this task's batches
        are images rather than tiles so a dropped one costs a single example.
        """
        model_spec = self._model_spec(model_name)
        base = self.dataset(model_name, split, augment=augment)
        training = split == "train"
        return make_classification_loader(
            base,
            batch_size=model_spec.batch_size,
            shuffle=training and self.spec.data.shuffle,
            num_workers=self.num_workers,
            drop_last=training and len(base) > model_spec.batch_size,
        )

    def balance(self, split: str = "train") -> ClassBalance:
        """How many images each class has in a split, and how lopsided that is.

        Public, and on the engine rather than only on the dataset, because it is the number
        every decision about this task's metrics turned on - and the person who most needs
        it is the one who has just written a specification and not yet run anything.
        """
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
                device=self.device,
            )
            validation = (
                self.loader(model_spec.name, "validation")
                if "validation" in self._samples
                else None
            )
            history = trainer.fit(
                self.loader(model_spec.name, "train"),
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

    def predict_labels(self, model_name: str | None = None, split: str = "test") -> list[str]:
        """The class names, because a name is what a reader wanted.

        Read off `data.classes`, whose **order is the class index** - so this is the one
        place where the engine knows the right names and a caller passing their own could
        be wrong without anything saying so.
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
        """
        name = self._resolve_model(model_name)
        probabilities = self.predict(name, split)
        classes = self.spec.data.classes
        samples = self._samples[split]
        rows = []
        for sample, row in zip(samples, probabilities, strict=True):
            chosen = int(row.argmax())
            rows.append(
                {
                    "key": sample.key,
                    "label": sample.label,
                    "predicted": classes[chosen],
                    "probability": float(row[chosen]),
                    **{
                        f"p_{klass}": float(value)
                        for klass, value in zip(classes, row, strict=True)
                    },
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
