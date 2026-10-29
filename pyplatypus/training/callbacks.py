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
import time
import warnings
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
    Ema as EmaSpec,
)
from pyplatypus.spec.components import (
    LinearAnnealing as LinearAnnealingSpec,
)
from pyplatypus.spec.components import (
    ModelCheckpoint as ModelCheckpointSpec,
)
from pyplatypus.spec.components import (
    ReduceLrOnPlateau as ReduceLrOnPlateauSpec,
)
from pyplatypus.spec.components import (
    StepDecay as StepDecaySpec,
)
from pyplatypus.spec.components import (
    StopAfter as StopAfterSpec,
)
from pyplatypus.spec.components import (
    Swa as SwaSpec,
)
from pyplatypus.spec.components import (
    TerminateOnCollapse as TerminateOnCollapseSpec,
)
from pyplatypus.spec.components import (
    TerminateOnNaN as TerminateOnNaNSpec,
)
from pyplatypus.weights import save_state

#: The modules whose running statistics an averaged weight tensor cannot carry honestly.
_BATCH_NORM = (nn.modules.batchnorm._BatchNorm,)


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
    #: Optimiser steps taken since training began, counted across epochs rather than
    #: within one. What a per-step average is a function of: a decay of 0.999 means 0.999
    #: per *step*, and over 60 epochs of 200 batches that is 12,000 updates rather than
    #: 60. Only the training pass advances it - a validation batch is not a step.
    step: int = 0
    logs: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    stop_reason: str | None = None
    #: What the run is training on. Only `swa` needs it, and it needs it for the one step
    #: that is easy to leave out: averaged weights carry batch-normalisation statistics
    #: from whichever epoch happened to be last, and recomputing them takes a pass over
    #: the training data. Without that the averaged model scores far worse than it should
    #: and nothing says why, so a callback that cannot reach the data cannot do SWA
    #: honestly.
    train_loader: Any = None


class Callback:
    """Four moments a callback may act on, and the fourth one is deliberately quiet.

    `on_batch_end` fires after every optimiser **step**, on the training pass only, with
    `state.step` counting steps since training began. It exists because a per-step average
    is a different estimator from a per-epoch one and cannot be approximated by it: a decay
    of 0.999 is 0.999 *per step*, which over 60 epochs of 200 batches is 12,000 updates.
    Announcing it per epoch would have meant offering `ema` under a name whose published
    values mean something else.

    **It cannot stop the run**, which is why it returns nothing. A run that stopped mid
    epoch would leave a record with training numbers and no validation ones, and every
    reader of the history - `best()`, the callbacks that watch it, `training_history()` in
    R - would have to learn that a row may be half a row. Stopping stays a decision taken
    where an epoch's numbers are complete; `stop_after`'s budget is therefore the budget
    plus one epoch, and says so.
    """

    def on_train_begin(self, state: TrainingState) -> None: ...

    def on_batch_end(self, state: TrainingState) -> None: ...

    def on_epoch_end(self, state: TrainingState) -> bool:
        """Return True to stop training."""
        return False

    def on_train_end(self, state: TrainingState) -> None: ...


def first_epoch(start: float | None, start_epoch: int | None, total: int) -> int:
    """The first epoch at which a callback may act, from whichever of the two was given.

    A fraction is resolved against the run's length, so `start=0.1` over 60 epochs begins
    at 7 - the first epoch *after* the first tenth, because an epoch is finished or it has
    not happened. Neither given is epoch 1, which is what every callback here did before
    `0.8.0a34`.
    """
    if start_epoch is not None:
        return start_epoch
    if start is not None:
        return int(total * start) + 1
    return 1


def ask_to_stop(callbacks: list[Callback], state: TrainingState) -> bool:
    """Whether any callback wants the run to end, having asked **every one of them**.

    One definition with three callers - the segmentation, detection and classification
    trainers - because the three had a copy each and every copy was
    `any(callback.on_epoch_end(state) for callback in self.callbacks)`. `any()` over a
    generator short-circuits, so the callbacks listed after the first one that answered
    `True` never saw the epoch the run stopped on. `pyplatypus#334`.

    What that cost, measured on the issue: a `csv_logger` written after a stopper lost the
    run's final row, and - worse, because the schema recommends it in as many words -
    `early_stopping(restore_best=True)` written after `stop_after` restored the
    second-best epoch instead of the best. `stop_after`'s own description says to pair it
    with `restore_best` *"if what the run produced matters"*, and writing the pair in that
    order is what broke it.

    A list is the whole fix. It is a function rather than a line in three files so that
    the next trainer cannot reintroduce the generator, and so that one test holds all
    three - the shape `labels_in_test_split` already has for the question of whether a
    split carries labels.
    """
    asked = [callback.on_epoch_end(state) for callback in callbacks]
    return any(asked)


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
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        start: float | None = None,
        start_epoch: int | None = None,
    ):
        super().__init__(monitor, min_delta)
        self.patience = patience
        self.restore_best = restore_best
        self.start = start
        self.start_epoch = start_epoch
        self.waited = 0
        self._from = 1
        self._best_weights: dict[str, torch.Tensor] | None = None

    def on_train_begin(self, state: TrainingState) -> None:
        self._from = first_epoch(self.start, self.start_epoch, state.total_epochs)

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
        # The best is recorded above whatever the warm-up says; only the counting waits.
        # An epoch before the warm-up ends is still the best weights this run has if it
        # ends for some other reason, and discarding it would lose a real answer.
        if state.epoch < self._from:
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
    def __init__(
        self,
        path: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        when: dict[str, float] | None = None,
    ):
        super().__init__(monitor)
        self.path = Path(path)
        self.save_best_only = save_best_only
        self.when = when or {}
        self.saved_epoch: int | None = None

    def on_train_begin(self, state: TrainingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def conditions_hold(self, logs: dict[str, float]) -> bool:
        """Every threshold, with its direction derived the way `monitor`'s is.

        Anything ending in `loss` has to be below its threshold and everything else above
        it, so there is no `above`/`below` field to write backwards.
        """
        return all(
            logs[key] <= limit if better_is_lower(key) else logs[key] >= limit
            for key, limit in self.when.items()
        )

    def on_epoch_end(self, state: TrainingState) -> bool:
        value = state.logs[self.monitor]
        if self.save_best_only and not self.improved(value):
            return False
        # An extra gate rather than a replacement: with both, a file is written when the
        # watched number improves *and* every threshold holds.
        if not self.conditions_hold(state.logs):
            return False
        self.best = value
        self.saved_epoch = state.epoch
        save_state(state.model.state_dict(), self.path)
        return False

    def on_train_end(self, state: TrainingState) -> None:
        """Say when nothing was written, because silence here is indistinguishable from
        a callback nobody configured - and a run that kept no weights is worth knowing
        about before somebody goes looking for the file."""
        if self.when and self.saved_epoch is None:
            conditions = ", ".join(f"{key} {limit}" for key, limit in self.when.items())
            warnings.warn(
                f"model_checkpoint wrote nothing: its conditions ({conditions}) never all "
                f"held, so there is no file at {self.path}. The run is over and its "
                "weights are whatever the last epoch produced.",
                stacklevel=2,
            )


class ReduceLrOnPlateau(_Watcher):
    def __init__(
        self, monitor: str = "val_loss", factor: float = 0.1, patience: int = 5, min_lr: float = 0.0
    ):
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


class _Schedule(Callback):
    """A learning rate that is a function of the epoch, not of how the run is going.

    The horizon and the per-group initial rates are the same for every schedule, and so is
    the reason each group keeps its own: `encoder_learning_rate` would otherwise be
    flattened at the first epoch, invisibly, because the history records one
    `learning_rate` - the first group's.
    """

    kind = "schedule"

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
                f"{type(self).__name__} needs to know how many epochs the run will take; "
                "the trainer did not say, and `epochs` was not given on the callback"
            )

    def factor_at(self, epoch: int) -> float:
        """How much of the way from `min_lr` to the initial rate this epoch sits."""
        raise NotImplementedError

    def on_epoch_end(self, state: TrainingState) -> bool:
        factor = self.factor_at(state.epoch)
        for group, initial in zip(state.optimizer.param_groups, self._initial, strict=False):
            group["lr"] = self.min_lr + (initial - self.min_lr) * factor
        return False


class CosineAnnealing(_Schedule):
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
    puts things that change the rate. It is **not** combinable with
    `reduce_lr_on_plateau`, and the sentence here used to say the opposite: a schedule
    recomputes the rate from the initial value every epoch, so it erases the plateau's
    cuts. The spec refuses the pair rather than letting one silently win.
    """

    def factor_at(self, epoch: int) -> float:
        # Clamped, so a run that goes past the horizon - `fit(epochs=...)` overriding the
        # spec - holds at min_lr instead of climbing back up the far side of the cosine.
        progress = min(epoch / self._horizon, 1.0)
        return (1 + math.cos(math.pi * progress)) / 2


class LinearAnnealing(_Schedule):
    def factor_at(self, epoch: int) -> float:
        # Clamped for the same reason the cosine is: a run past the horizon holds at
        # min_lr rather than walking off the end of the line into negative rates.
        return 1.0 - min(epoch / self._horizon, 1.0)


class StepDecay(Callback):
    """Multiply by `factor` at fixed epochs, computed from the initial rate each time.

    Computed rather than accumulated - `initial * factor ** drops` - so replaying an epoch
    cannot double a drop, and so the schedule says the same thing whatever else wrote to
    the learning rate in between.
    """

    kind = "schedule"

    def __init__(
        self,
        factor: float = 0.1,
        every: int | None = None,
        milestones: list[int] | None = None,
        min_lr: float = 0.0,
    ):
        self.factor = factor
        self.every = every
        self.milestones = milestones
        self.min_lr = min_lr
        self._initial: list[float] = []

    def on_train_begin(self, state: TrainingState) -> None:
        self._initial = [group["lr"] for group in state.optimizer.param_groups]

    def drops_by(self, epoch: int) -> int:
        if self.milestones is not None:
            return sum(1 for milestone in self.milestones if milestone <= epoch)
        return epoch // self.every

    def on_epoch_end(self, state: TrainingState) -> bool:
        drops = self.drops_by(state.epoch)
        for group, initial in zip(state.optimizer.param_groups, self._initial, strict=False):
            group["lr"] = max(initial * self.factor**drops, self.min_lr)
        return False


class Swa(Callback):
    """Average the weights over the last part of training instead of taking the final ones.

    Stochastic weight averaging: past `start`, every epoch's weights are folded into a
    running average, and at the end that average becomes the model. The claim is that a
    point in the middle of a flat region generalises better than whichever corner of it the
    last epoch happened to stop in.

    **The step that is easy to leave out is the one that decides whether it works.** An
    averaged weight tensor has batch-normalisation running statistics belonging to no
    particular epoch - they were not averaged, they were inherited - so the averaged model
    is evaluated with the wrong normalisation unless the statistics are recomputed by a pass
    over the training data. Omitted, the result is a model that scores far worse than it
    should with nothing to say why, which is why this callback asks the state for the
    training loader rather than doing without.

    `learning_rate` holds the rate constant once averaging begins, which is the recipe:
    averaging is only worth anything if the weights are still moving, and a rate that has
    decayed to nothing produces a set of nearly identical snapshots. For the same reason a
    run cannot ask for `swa` and `cosine_annealing` at once - the specification refuses it.
    """

    def __init__(
        self,
        start: float = 0.75,
        learning_rate: float | None = None,
        start_epoch: int | None = None,
    ):
        self.start = start
        self.start_epoch = start_epoch
        self.learning_rate = learning_rate
        self._averaged = None
        self._first: int | None = None
        self._folded = 0

    def on_train_begin(self, state: TrainingState) -> None:
        from torch.optim.swa_utils import AveragedModel

        self._averaged = AveragedModel(state.model)
        # Counting from 1 like `epoch` does, and at least the last epoch: `start=1.0` means
        # "average the final weights", which is a no-op rather than a run with nothing
        # averaged at all.
        horizon = max(1, state.total_epochs)
        # `start_epoch` says the same thing in epochs, and is clamped to the run's length
        # for the same reason the fraction is: a start past the end would fold nothing,
        # and folding the last epoch alone is the honest reading of "begin at the end".
        # Through `first_epoch`, which is where every other callback's `start` is
        # resolved. This read `round(horizon * self.start) + 1` until `pyplatypus#335`,
        # and `round` is half-to-even, so the two disagreed by an epoch whenever
        # `horizon * start` landed on a half with an odd integer part - (10, 0.75) gave 8
        # here and 9 there. The schema says of `swa.start_epoch` that "it is the same
        # concept `early_stopping` names with these two words", so there is one rule.
        self._first = min(horizon, max(1, first_epoch(self.start, self.start_epoch, horizon)))
        self._folded = 0

    def on_epoch_end(self, state: TrainingState) -> bool:
        if self._averaged is None or state.epoch < self._first:
            return False
        self._averaged.update_parameters(state.model)
        self._folded += 1
        if self.learning_rate is not None:
            for group in state.optimizer.param_groups:
                group["lr"] = self.learning_rate
        return False

    def on_train_end(self, state: TrainingState) -> None:
        """The average becomes the model, and then its statistics are made true again."""
        from torch.optim.swa_utils import update_bn

        if self._averaged is None or self._folded == 0:
            # Early stopping can end a run before averaging ever began. Leaving the final
            # weights alone is right: there is no average, and substituting one epoch's
            # weights for "the average" would be a lie about what happened.
            return

        state.model.load_state_dict(self._averaged.module.state_dict())
        # Into the last record, not `state.logs`: by the time `on_train_end` runs the
        # per-epoch logs have already been written, so a number put there reaches nobody.
        # Found by looking for it in `training_history()` and not finding it.
        if state.history:
            state.history[-1]["swa_epochs_averaged"] = float(self._folded)

        if state.train_loader is None:
            return
        if not any(isinstance(m, _BATCH_NORM) for m in state.model.modules()):
            return
        update_bn(state.train_loader, state.model, device=next(state.model.parameters()).device)


class StopAfter(Callback):
    """End the run once a wall-clock budget is spent.

    Checked at the end of an epoch, so the real limit is the budget plus one epoch. That
    is said in the field's description rather than fixed with a per-batch hook, because
    this package's callbacks do not see a step and adding that for a clock would be a
    large change for a small gain.
    """

    def __init__(self, hours: float):
        self.hours = hours
        self._started = 0.0

    def on_train_begin(self, state: TrainingState) -> None:
        self._started = time.perf_counter()

    def on_epoch_end(self, state: TrainingState) -> bool:
        spent = (time.perf_counter() - self._started) / 3600
        if spent < self.hours:
            return False
        state.stop_reason = (
            f"stop_after: {spent:.2f} hours spent of a {self.hours:g} hour budget, "
            f"at epoch {state.epoch}. The clock ran out; the run did not converge."
        )
        return True


class TerminateOnCollapse(Callback):
    """Stop once the prediction has held no foreground for `patience` epochs.

    Not a `_Watcher`: this is a threshold rather than an improvement, and the direction is
    not derived from the name - `foreground` is neither a loss to minimise nor a score to
    maximise, it is a share that should simply not be zero.
    """

    def __init__(self, floor: float = 0.0001, patience: int = 3, monitor: str = "val_foreground"):
        self.floor = floor
        self.patience = patience
        self.monitor = monitor
        self.below = 0

    def on_epoch_end(self, state: TrainingState) -> bool:
        share = state.logs[self.monitor]
        if share > self.floor:
            self.below = 0
            return False
        self.below += 1
        if self.below < self.patience:
            return False
        state.stop_reason = (
            f"terminate_on_collapse: {self.monitor} has been at or below {self.floor:g} "
            f"for {self.patience} epochs - the model is predicting background everywhere, "
            "and the loss and Dice can both look reasonable while it does"
        )
        return True


class Ema(Callback):
    """A moving average of the weights, folded in after every step.

    `AveragedModel` with torch's own EMA function rather than an average written here, for
    the same reason `swa` uses it: the averaging rule is somebody else's arithmetic and
    this package has no opinion about it.

    `use_buffers=True` is the one difference from `swa` and it replaces a pass over the
    training data: a uniform average of weights has batch-normalisation statistics
    belonging to no epoch, so `swa` recomputes them at the end; an exponential average of
    the buffers alongside the weights carries statistics of its own.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self._averaged = None
        self._folded = 0

    def on_train_begin(self, state: TrainingState) -> None:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        self._averaged = AveragedModel(
            state.model, multi_avg_fn=get_ema_multi_avg_fn(self.decay), use_buffers=True
        )
        self._folded = 0

    def on_batch_end(self, state: TrainingState) -> None:
        if self._averaged is None:
            return
        self._averaged.update_parameters(state.model)
        self._folded += 1

    def on_train_end(self, state: TrainingState) -> None:
        if self._averaged is None or self._folded == 0:
            # A run that stopped before taking a step has no average, and substituting the
            # initial weights for one would be a lie about what happened - the same
            # reasoning `swa` applies to a run that ended before averaging began.
            return
        state.model.load_state_dict(self._averaged.module.state_dict())
        if state.history:
            state.history[-1]["ema_steps"] = float(self._folded)


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
    EarlyStoppingSpec: lambda s: EarlyStopping(
        s.monitor, s.patience, s.min_delta, s.restore_best, s.start, s.start_epoch
    ),
    ModelCheckpointSpec: lambda s: ModelCheckpoint(s.path, s.monitor, s.save_best_only, s.when),
    StopAfterSpec: lambda s: StopAfter(s.hours),
    ReduceLrOnPlateauSpec: lambda s: ReduceLrOnPlateau(s.monitor, s.factor, s.patience, s.min_lr),
    CosineAnnealingSpec: lambda s: CosineAnnealing(s.min_lr, s.epochs),
    LinearAnnealingSpec: lambda s: LinearAnnealing(s.min_lr, s.epochs),
    StepDecaySpec: lambda s: StepDecay(s.factor, s.every, s.milestones, s.min_lr),
    EmaSpec: lambda s: Ema(s.decay),
    SwaSpec: lambda s: Swa(s.start, s.learning_rate, s.start_epoch),
    CsvLoggerSpec: lambda s: CsvLogger(s.path),
    TerminateOnCollapseSpec: lambda s: TerminateOnCollapse(s.floor, s.patience, s.monitor),
    TerminateOnNaNSpec: lambda s: TerminateOnNaN(),
}


def _replaces_weights_last(callback: Callback) -> int:
    """Where a callback's `on_train_end` sits in `finish_training`. Lower runs first.

    THREE callbacks replace the model's weights in `on_train_end`, and the trainer calls
    them in list order, so the last one to run decides what the run produced.
    `EarlyStopping.restore_best` loads the best epoch's snapshot, `Ema` loads its average
    and `Swa` loads its own. `ModelSpec.one_averager` refuses `ema` beside `swa` on exactly
    that argument - *"both replace the model when training ends ... in whichever order the
    list happened to have them"* - and `restore_best` is the third, which neither the
    refusal nor this ordering accounted for until `pyplatypus#333`.

    Measured there, on one run with only the order changed: `[ema, early_stopping]` ended
    on the raw best epoch and `[early_stopping, ema]` on the average, 0.578 apart in the
    largest weight, while the history reported `ema_steps = 8` in both.

    **The averager wins**, which is the same answer `swa` already got here and for a
    related reason: somebody who configured averaging asked for the averaged model, and
    `restore_best` is a statement about which *epoch* was best rather than about which
    weights the run should hand back. A refusal was the alternative and is the wrong shape:
    `restore_best` defaults to `True`, so refusing the pair would reject configurations
    whose authors never knowingly asked for it.

    `swa` is last of the three, which also happens to be where `build_callbacks` puts it -
    for its own and unrelated reason, the learning rate it pins during the epochs.
    """
    if isinstance(callback, Swa):
        return 3
    if isinstance(callback, Ema):
        return 2
    if isinstance(callback, EarlyStopping) and callback.restore_best:
        return 1
    return 0


def finish_training(callbacks: list[Callback], state: TrainingState) -> None:
    """Run every callback's `on_train_end`, with the weight-replacers last and in order.

    One definition with three callers, for `ask_to_stop`'s reason and one more: this
    guarantee has to hold however the list arrived, and a caller may hand a `Trainer` its
    own `callbacks=` list that never went through `build_callbacks`. Ordering at the point
    of building would have left that route exactly as `pyplatypus#333` found it - which is
    what the test for this caught, on the first version of the fix.
    """
    for callback in sorted(callbacks, key=_replaces_weights_last):
        callback.on_train_end(state)


def build_callbacks(specs: list[CallbackSpec]) -> list[Callback]:
    """Built in the order given, except that the weight-replacers are moved to the end.

        Two callbacks may write the learning rate in one epoch: a schedule, which computes it
        from the optimiser's initial value, and `swa`, which pins it to a constant once
        averaging has begun. Which one the epoch ends with decided itself by where they
        happened to sit in the list, and nothing said so - so the order is fixed here instead.
        `swa` wins, because a constant tail rate is the whole point of configuring it.

        The spec refuses the other pairs outright: two schedules, or a schedule beside
        `reduce_lr_on_plateau`, which a schedule erases every epoch.

    **Which weights the run ends on is a different question and is not settled here**, even
        though it also used to be decided by list position. It is settled in `finish_training`,
        because it is about the order `on_train_end` runs in rather than the order the epochs
        see - and because a caller may build a `Trainer` with a `callbacks=` list of its own,
        which never passes through this function. `pyplatypus#333`.
    """
    built = [_BUILDERS[type(spec)](spec) for spec in specs]
    # Stable, so everything else keeps the order it was written in.
    return sorted(built, key=lambda callback: isinstance(callback, Swa))
