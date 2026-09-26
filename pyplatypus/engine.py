"""Many models, one spec.

This is what `platypus_fit(spec)` calls. Each model gets its own data pipeline, because
each may want a different input size or a different tiling, and its own augmentation,
because that is part of the experiment. Validation never gets augmented - measuring a
model on distorted data measures the distortion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyplatypus.data.augmentation import build_augmenter
from pyplatypus.data.dataset import SegmentationDataset
from pyplatypus.data.paths import Sample, discover
from pyplatypus.data.splits import group_of
from pyplatypus.errors import PlatypusError
from pyplatypus.models import build_model
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.spec.spec import PlatypusSpec
from pyplatypus.training.torch_data import make_loader
from pyplatypus.training.trainer import History, Trainer


class EngineError(PlatypusError):
    kind = "engine_error"


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


class Engine:
    def __init__(self, spec: PlatypusSpec, *, device: str | None = None,
                 num_workers: int = 0, strict_data: bool = True):
        if spec.rank != 2:
            raise EngineError(
                f"this engine trains 2D models; the spec is {spec.rank}D. "
                "The spec and the model builder already handle volumes - the data "
                "pipeline is what v0.1 stops at."
            )
        self.spec = spec
        self.device = device
        self.num_workers = num_workers
        self.runs: dict[str, ModelRun] = {}

        self._samples: dict[str, tuple[Sample, ...]] = {
            "train": discover(spec.data.train_path, spec.data, strict=strict_data).samples,
            "validation": discover(spec.data.validation_path, spec.data,
                                   strict=strict_data).samples,
        }
        if spec.data.test_path:
            self._samples["test"] = discover(spec.data.test_path, spec.data,
                                             only_images=True, strict=strict_data).samples

    # ------------------------------------------------------------------ data
    def dataset(self, model: SegmentationModel, split: str, *, augmented: bool = False,
                only_images: bool = False) -> SegmentationDataset:
        if split not in self._samples:
            raise EngineError(
                f"no '{split}' data in this spec; available: "
                f"{', '.join(sorted(self._samples))}"
            )
        augmenter = build_augmenter(model.augmentation, model.rank) if augmented else None
        return SegmentationDataset(
            self._samples[split], model, self.spec.data,
            augmenter=augmenter, only_images=only_images,
        )

    def loader(self, model: SegmentationModel, split: str, *, augmented: bool = False,
               shuffle: bool = False, only_images: bool = False):
        return make_loader(
            self.dataset(model, split, augmented=augmented, only_images=only_images),
            batch_size=model.batch_size, shuffle=shuffle, num_workers=self.num_workers,
        )

    # ----------------------------------------------------------------- weights
    @staticmethod
    def _load_weights(model: torch.nn.Module, reference: str) -> None:
        path = Path(reference)
        if not path.exists():
            raise EngineError(
                f"'{reference}' is not a file. Named weights from the registry are not "
                "wired up yet; give a path to a local checkpoint for now."
            )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    # -------------------------------------------------------------------- fit
    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        """Train every model the spec asks for, in order."""
        for model_spec in self.spec.models:
            if verbose:
                print(f"\n=== {model_spec.name} ({model_spec.architecture.value}) ===")

            network = build_model(model_spec)
            if model_spec.weights:
                self._load_weights(network, model_spec.weights)

            trainer = Trainer(network, model_spec, device=self.device)
            run = ModelRun(name=model_spec.name, spec=model_spec,
                           model=trainer.model, trainer=trainer)

            if model_spec.fit:
                run.history = trainer.fit(
                    self.loader(model_spec, "train", augmented=True,
                                shuffle=self.spec.data.shuffle),
                    self.loader(model_spec, "validation"),
                    verbose=verbose,
                )
                run.trained = True
            self.runs[model_spec.name] = run

        return {name: run.history for name, run in self.runs.items()}

    # --------------------------------------------------------------- evaluate
    def evaluate(self, split: str = "validation") -> list[dict[str, Any]]:
        """One row per model: the comparison table the whole multi-model idea is for."""
        if not self.runs:
            raise EngineError("nothing has been trained or loaded yet; call fit() first")

        table = []
        for run in self.runs.values():
            scores = run.trainer.evaluate(self.loader(run.spec, split))
            best = run.history.best("val_loss")
            table.append({
                "model": run.name,
                "architecture": run.spec.architecture.value,
                # Named, because a loss column is only comparable between models that
                # were trained on the same one - see best_model().
                "loss_function": run.spec.loss.name,
                "parameters": run.parameters,
                "epochs_run": len(run.history),
                "best_epoch": best["epoch"] if best else None,
                **{key.removeprefix("val_"): value for key, value in scores.items()},
            })
        return table

    def evaluate_cases(self, model_name: str, split: str = "validation", *,
                       group_by: str | None = None) -> list[dict[str, Any]]:
        """One row per case instead of one row per model.

        The comparison table answers "which model"; this answers "on whom does it fail",
        which is the question a clinician asks first and the one a single mean cannot
        answer. With `group_by`, each row also carries the group - the patient, usually -
        so the rows can be summarised per patient rather than per slice.
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

        dataset = self.dataset(run.spec, split)
        tiles = dataset.tiles_per_sample
        cases = [
            group_of(sample.key, group_by) if group_by else sample.key
            for sample in dataset.samples
            for _ in range(tiles)
        ]
        # Never shuffled: the case names line up with the order examples arrive in.
        loader = self.loader(run.spec, split, shuffle=False)
        rows = run.trainer.score_cases(loader, cases)
        key = "group" if group_by else "case"
        return [{key: row.pop("case"), **row} for row in rows]

    # ---------------------------------------------------------------- predict
    def predict(self, model_name: str, split: str = "test") -> np.ndarray:
        """Class probabilities, channels-last, one array per source image.

        Tiles are reassembled, so an image that went in at 2048x1536 comes back at
        2048x1536 rather than as 24 unrelated pieces.
        """
        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        only_images = split == "test" and "test" in self._samples
        # Never shuffle: stitching depends on tiles arriving in the order they were cut.
        loader = self.loader(run.spec, split, shuffle=False, only_images=only_images)
        return run.trainer.predict(loader)

    def best_model(self, key: str = "dice", split: str = "validation") -> str:
        """Rank the models on one column.

        Refuses to rank on `loss` when the models were not trained on the same one. Two
        losses are two different scales: a Focal-Tversky of 0.05 is not better than a
        CCE-Dice of 0.14, it is not even the same question. Metrics are comparable
        because they measure the mask, not the objective.
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


def summarise_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn per-case rows into the distribution, one row per metric.

    What a paper reports: not a single Dice, but its mean, its spread and its worst case.
    The minimum is the one nobody publishes and everybody should - a model averaging 0.86
    that scores 0.11 on some patient has a failure mode, and the mean is where it hides.

    Standard deviation is the sample one (n-1), and is None for a single case, because the
    spread of one number is not zero, it is unknown.
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
        out.append({
            "metric": metric,
            "n": int(values.size),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            f"worst_{label}": worst[label],
        })
    return out
