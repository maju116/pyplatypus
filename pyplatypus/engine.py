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
from pyplatypus.data.dataset import SegmentationDataset, _is_series
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
    def _load_weights(model: torch.nn.Module, reference: str,
                      spec: SegmentationModel) -> dict | None:
        """A registry name, a Hub reference, or a local path - see `pyplatypus.weights`."""
        from pyplatypus.weights import load_into

        return load_into(model, reference, spec)

    def export_weights(self, model_name: str, path: str | Path, **extra) -> Path:
        """Write one trained model's weights, ready to publish or to load again later.

        safetensors plus a sidecar recording what the weights are for. Anything passed as
        `extra` joins the sidecar, which is where the data they were trained on and its licence
        belong - a weights file whose provenance is only in somebody's memory cannot be used by
        anybody else.
        """
        from pyplatypus.weights import export_weights

        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        return export_weights(run.model, run.spec, path, extra=extra or None)

    # -------------------------------------------------------------------- fit
    def fit(self, *, verbose: bool = False) -> dict[str, History]:
        """Train every model the spec asks for, in order."""
        for model_spec in self.spec.models:
            if verbose:
                print(f"\n=== {model_spec.name} ({model_spec.architecture.value}) ===")

            network = build_model(model_spec)
            if model_spec.weights:
                self._load_weights(network, model_spec.weights, model_spec)

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
        """
        if space not in ("model", "source"):
            raise EngineError(f"space is 'model' or 'source', got '{space}'")

        run = self.runs.get(model_name)
        if run is None:
            known = ", ".join(self.runs) or "none"
            raise EngineError(f"no model called '{model_name}'; trained so far: {known}")
        only_images = split == "test" and "test" in self._samples
        # Never shuffle: stitching depends on tiles arriving in the order they were cut.
        loader = self.loader(run.spec, split, shuffle=False, only_images=only_images)
        predictions = run.trainer.predict(loader)

        if space == "model":
            return predictions
        samples = self._samples[split]
        return [
            self._to_source_space(predictions[index], sample, run.spec)
            for index, sample in enumerate(samples)
        ]

    def _to_source_space(self, prediction: np.ndarray, sample: Sample,
                         model: SegmentationModel) -> np.ndarray:
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
                np.zeros((*shape, 1), dtype=np.float32), spacing,
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
            ordered = match_channels(sample.images, self.spec.data.channels_from,
                                     key=sample.key)
            return spatial_shape(ordered[0])
        return spatial_shape(sample.images[0])

    def _source_spacing(self, sample: Sample):
        from pyplatypus.data.channels import match_channels
        from pyplatypus.data.dicom_series import series_spacing
        from pyplatypus.data.volumes import volume_spacing

        if _is_series(sample.images):
            return series_spacing(list(sample.images))
        if self.spec.data.channels_from is not None and len(sample.images) > 1:
            ordered = match_channels(sample.images, self.spec.data.channels_from,
                                     key=sample.key)
            return volume_spacing(ordered[0])
        return volume_spacing(sample.images[0])

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
