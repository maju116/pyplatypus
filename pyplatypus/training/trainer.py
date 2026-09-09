"""Training one model.

Deep supervision is handled here rather than in the losses: the model hands back one
prediction per depth, the loss is averaged over all of them, and the metrics are scored
on the final one only - which is the prediction the user will actually receive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pyplatypus.data.images import stitch
from pyplatypus.objectives import build_loss, build_metrics
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training.callbacks import Callback, TrainingState, build_callbacks
from pyplatypus.training.optimizers import build_optimizer


@dataclass
class History:
    """One row per epoch. Plain data, so it crosses into R as a data.frame."""

    records: list[dict[str, float]] = field(default_factory=list)
    stop_reason: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def columns(self) -> list[str]:
        return list(self.records[0]) if self.records else []

    def best(self, key: str = "val_loss") -> dict[str, float] | None:
        if not self.records or key not in self.records[0]:
            return None
        lower_is_better = key.endswith("loss")
        return (min if lower_is_better else max)(self.records, key=lambda r: r[key])

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records, "stop_reason": self.stop_reason}


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model: nn.Module, spec: SegmentationModel, *,
                 device: str | None = None, callbacks: list[Callback] | None = None):
        self.spec = spec
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = build_loss(spec.loss)
        self.metrics = build_metrics(spec.metrics)
        self.optimizer = build_optimizer(spec.optimizer, self.model.parameters())
        self.callbacks = callbacks if callbacks is not None else build_callbacks(spec.callbacks)

    def _loss_and_final(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        out = self.model(batch_x)
        if isinstance(out, tuple):
            # Deep supervision: every depth is trained, the deepest is reported.
            loss = torch.stack([self.loss_fn(o, batch_y) for o in out]).mean()
            return loss, out[-1]
        return self.loss_fn(out, batch_y), out

    def _run_epoch(self, loader: DataLoader, *, train: bool, prefix: str
                   ) -> dict[str, float]:
        self.model.train(train)
        totals: dict[str, float] = {f"{prefix}_loss": 0.0}
        totals.update({f"{prefix}_{name}": 0.0 for name in self.metrics})
        batches = 0

        with torch.set_grad_enabled(train):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                loss, final = self._loss_and_final(batch_x, batch_y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                totals[f"{prefix}_loss"] += float(loss)
                for name, metric in self.metrics.items():
                    totals[f"{prefix}_{name}"] += float(metric(final, batch_y))
                batches += 1

        if batches == 0:
            raise ValueError(f"the {prefix} loader produced no batches")
        return {key: value / batches for key, value in totals.items()}

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader | None = None,
            *, epochs: int | None = None, verbose: bool = False) -> History:
        epochs = epochs if epochs is not None else self.spec.epochs
        state = TrainingState(model=self.model, optimizer=self.optimizer)
        history = History()

        for callback in self.callbacks:
            callback.on_train_begin(state)

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            logs = self._run_epoch(train_loader, train=True, prefix="train")
            if validation_loader is not None:
                logs.update(self._run_epoch(validation_loader, train=False, prefix="val"))
            logs["seconds"] = time.perf_counter() - started
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

            state.epoch = epoch
            state.logs = logs
            record = {"epoch": epoch, **logs}
            history.records.append(record)
            state.history = history.records
            if verbose:
                shown = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "seconds")
                print(f"epoch {epoch:>3}  {shown}  ({logs['seconds']:.1f}s)")

            if any(callback.on_epoch_end(state) for callback in self.callbacks):
                history.stop_reason = state.stop_reason
                break

        for callback in self.callbacks:
            callback.on_train_end(state)
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return self._run_epoch(loader, train=False, prefix="val")

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Class probabilities for every example, channels-last, tiles reassembled.

        The old package stopped at the tiles. Cutting an HD image into a grid is only
        useful if what comes back is the same size as what went in, so if the model tiles,
        consecutive tiles are stitched into one mask per source image here.
        """
        self.model.eval()
        chunks = []
        for batch in loader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = self.model(batch_x.to(self.device))
            if isinstance(out, tuple):
                out = out[-1]
            probabilities = out.softmax(dim=1).cpu().numpy()
            chunks.append(np.moveaxis(probabilities, 1, -1))

        predictions = np.concatenate(chunks, axis=0)
        splits = self.spec.splits
        if splits is None:
            return predictions

        per_image = self.spec.tiles_per_image
        if len(predictions) % per_image:
            raise ValueError(
                f"got {len(predictions)} tiles, which is not a whole number of images "
                f"at {per_image} tiles each; the loader must not drop or shuffle them"
            )
        return np.stack([
            stitch(predictions[i:i + per_image], splits)
            for i in range(0, len(predictions), per_image)
        ])
