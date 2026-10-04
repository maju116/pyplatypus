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
    CosineAnnealing as CosineAnnealingSpec,
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
    #: How many epochs the run will take at most. A schedule that is a function of
    #: progress needs the horizon, and nothing else in the state implies it: `epoch`
    #: counts up and stops wherever early stopping stops it.
    total_epochs: int = 0
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


class CosineAnnealing(Callback):
    """Decay the learning rate along a cosine from its initial value to `min_lr`.

    Wanted because the measured BCCD run used it and the specification could not say so.
    What it is worth on that dataset was then measured, twice: adding it moved the test
    `mAP@[.50:.95]` from 0.4700 to 0.4761, and `mAP@0.5` from 0.8479 to 0.8496. Both of
    those runs are single draws and neither was seeded, so some of that is the schedule
    and some is the draw - it is the honest size of the effect and not a clean one.

    Two details decide whether this is the same schedule as torch's `CosineAnnealingLR`
    rather than something that resembles it:

    * **each parameter group decays from its own initial rate.** With
      `encoder_learning_rate` the groups start at different rates on purpose, and a
      schedule that computed one rate for all of them would quietly undo that at the first
      epoch.
    * **it steps after an epoch, from the epoch just finished.** That is where
      `scheduler.step()` goes in an ordinary loop, so epoch 1 trains at the initial rate
      and epoch `total_epochs` trains at `min_lr`.

    It is a callback and not a field on the optimizer because that is where this package
    puts things that change the rate - `reduce_lr_on_plateau` is one too - and because a
    run may want both: a cosine floor with a plateau rescue on top is a legitimate recipe.
    """

    def __init__(self, min_lr: float = 0.0, epochs: int | None = None):
        self.min_lr = min_lr
        self.epochs = epochs
        self._initial: list[float] = []
        self._horizon = 0

    def on_train_begin(self, state: TrainingState) -> None:
        self._initial = [group["lr"] for group in state.optimizer.param_groups]
        self._horizon = self.epochs if self.epochs is not None else state.total_epochs
        if self._horizon < 1:
            raise ValueError(
                "cosine_annealing needs to know how many epochs the run will take; the "
                "trainer did not say, and `epochs` was not given on the callback"
            )

    def on_epoch_end(self, state: TrainingState) -> bool:
        # Clamped, so a run that goes past the horizon - `fit(epochs=...)` overriding the
        # spec - holds at min_lr instead of climbing back up the far side of the cosine.
        progress = min(state.epoch / self._horizon, 1.0)
        factor = (1 + math.cos(math.pi * progress)) / 2
        for group, initial in zip(state.optimizer.param_groups, self._initial):
            group["lr"] = self.min_lr + (initial - self.min_lr) * factor
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
    CosineAnnealingSpec: lambda s: CosineAnnealing(s.min_lr, s.epochs),
    CsvLoggerSpec: lambda s: CsvLogger(s.path),
    TerminateOnNaNSpec: lambda s: TerminateOnNaN(),
}


def build_callbacks(specs: list[CallbackSpec]) -> list[Callback]:
    return [_BUILDERS[type(spec)](spec) for spec in specs]
