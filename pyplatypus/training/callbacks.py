"""Callbacks, because torch ships none.

Each one watches a key in the epoch's logs. Which keys exist is decided by the model's
metrics, and the spec already refused any callback watching something this model will
never produce - so nothing here has to cope with a missing key.

Whether a number should go up or down is derived, not configured: losses fall, metrics
rise. One less thing to get backwards in a YAML file.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.components import (
    CallbackSpec,
)
from pyplatypus.spec.components import (
    CsvLogger as CsvLoggerSpec,
)
from pyplatypus.spec.components import (
    EarlyStopping as EarlyStoppingSpec,
)
from pyplatypus.spec.components import (
    ModelCheckpoint as ModelCheckpointSpec,
)
from pyplatypus.spec.components import (
    ReduceLrOnPlateau as ReduceLrOnPlateauSpec,
)
from pyplatypus.spec.components import (
    TerminateOnNaN as TerminateOnNaNSpec,
)


class TrainingStopped(PlatypusError):
    """Raised by nothing; a callback asks to stop by returning True."""

    kind = "training_stopped"


def better_is_lower(key: str) -> bool:
    """Losses fall, metrics rise. Derived from the name so nobody has to declare it."""
    return key.endswith("loss")


@dataclass
class TrainingState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    epoch: int = 0
    logs: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    stop_reason: str | None = None


class Callback:
    def on_train_begin(self, state: TrainingState) -> None: ...

    def on_epoch_end(self, state: TrainingState) -> bool:
        """Return True to stop training."""
        return False

    def on_train_end(self, state: TrainingState) -> None: ...


class _Watcher(Callback):
    def __init__(self, monitor: str, min_delta: float = 0.0):
        self.monitor = monitor
        self.lower_is_better = better_is_lower(monitor)
        self.min_delta = min_delta
        self.best = math.inf if self.lower_is_better else -math.inf

    def improved(self, value: float) -> bool:
        if self.lower_is_better:
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta


class EarlyStopping(_Watcher):
    def __init__(self, monitor: str = "val_loss", patience: int = 10,
                 min_delta: float = 0.0, restore_best: bool = True):
        super().__init__(monitor, min_delta)
        self.patience = patience
        self.restore_best = restore_best
        self.waited = 0
        self._best_weights: dict[str, torch.Tensor] | None = None

    def on_epoch_end(self, state: TrainingState) -> bool:
        value = state.logs[self.monitor]
        if self.improved(value):
            self.best = value
            self.waited = 0
            if self.restore_best:
                self._best_weights = {
                    k: v.detach().clone() for k, v in state.model.state_dict().items()
                }
            return False
        self.waited += 1
        if self.waited >= self.patience:
            state.stop_reason = (
                f"early stopping: {self.monitor} has not improved on {self.best:.5f} "
                f"for {self.patience} epochs"
            )
            return True
        return False

    def on_train_end(self, state: TrainingState) -> None:
        if self.restore_best and self._best_weights is not None:
            state.model.load_state_dict(self._best_weights)


class ModelCheckpoint(_Watcher):
    def __init__(self, path: str, monitor: str = "val_loss", save_best_only: bool = True):
        super().__init__(monitor)
        self.path = Path(path)
        self.save_best_only = save_best_only
        self.saved_epoch: int | None = None

    def on_train_begin(self, state: TrainingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, state: TrainingState) -> bool:
        value = state.logs[self.monitor]
        if self.save_best_only and not self.improved(value):
            return False
        self.best = value
        self.saved_epoch = state.epoch
        torch.save(state.model.state_dict(), self.path)
        return False


class ReduceLrOnPlateau(_Watcher):
    def __init__(self, monitor: str = "val_loss", factor: float = 0.1,
                 patience: int = 5, min_lr: float = 0.0):
        super().__init__(monitor)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.waited = 0

    def on_epoch_end(self, state: TrainingState) -> bool:
        value = state.logs[self.monitor]
        if self.improved(value):
            self.best = value
            self.waited = 0
            return False
        self.waited += 1
        if self.waited >= self.patience:
            self.waited = 0
            for group in state.optimizer.param_groups:
                group["lr"] = max(group["lr"] * self.factor, self.min_lr)
        return False


class CsvLogger(Callback):
    def __init__(self, path: str):
        self.path = Path(path)
        self._columns: list[str] | None = None

    def on_train_begin(self, state: TrainingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def on_epoch_end(self, state: TrainingState) -> bool:
        row = {"epoch": state.epoch, **state.logs}
        new = self._columns is None
        if new:
            self._columns = list(row)
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            if new:
                writer.writeheader()
            writer.writerow(row)
        return False


class TerminateOnNaN(Callback):
    """A NaN loss never recovers; carrying on just burns GPU hours."""

    def on_epoch_end(self, state: TrainingState) -> bool:
        for key, value in state.logs.items():
            if not math.isfinite(value):
                state.stop_reason = f"{key} became {value} at epoch {state.epoch}"
                return True
        return False


_BUILDERS: dict[type, Any] = {
    EarlyStoppingSpec: lambda s: EarlyStopping(s.monitor, s.patience, s.min_delta,
                                               s.restore_best),
    ModelCheckpointSpec: lambda s: ModelCheckpoint(s.path, s.monitor, s.save_best_only),
    ReduceLrOnPlateauSpec: lambda s: ReduceLrOnPlateau(s.monitor, s.factor, s.patience,
                                                       s.min_lr),
    CsvLoggerSpec: lambda s: CsvLogger(s.path),
    TerminateOnNaNSpec: lambda s: TerminateOnNaN(),
}


def build_callbacks(specs: list[CallbackSpec]) -> list[Callback]:
    return [_BUILDERS[type(spec)](spec) for spec in specs]
