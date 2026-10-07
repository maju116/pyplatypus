import csv

import pytest
import torch
from torch import nn

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
from pyplatypus.training.callbacks import TrainingState, better_is_lower, build_callbacks


@pytest.fixture
def state():
    model = nn.Linear(2, 2)
    return TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))


def run(callback, state, values, key="val_loss"):
    """Feed a sequence of epoch values, return the epoch it asked to stop on."""
    callback.on_train_begin(state)
    for epoch, value in enumerate(values, start=1):
        state.epoch, state.logs = epoch, {key: value}
        if callback.on_epoch_end(state):
            callback.on_train_end(state)
            return epoch
    callback.on_train_end(state)
    return None


def test_direction_is_derived_from_the_name():
    """Losses fall, metrics rise. Nobody has to declare it, so nobody can get it backwards."""
    assert better_is_lower("val_loss") and better_is_lower("train_loss")
    assert not better_is_lower("val_dice") and not better_is_lower("val_iou")


def test_early_stopping_waits_out_the_patience(state):
    callback = build_callbacks([EarlyStoppingSpec(patience=2)])[0]
    assert run(callback, state, [1.0, 0.9, 0.9, 0.9]) == 4


def test_early_stopping_resets_when_things_improve(state):
    callback = build_callbacks([EarlyStoppingSpec(patience=2)])[0]
    assert run(callback, state, [1.0, 0.9, 0.9, 0.5, 0.5, 0.5]) == 6


def test_early_stopping_maximises_a_metric(state):
    """The same callback, watching a Dice, has to want it to go up."""
    callback = build_callbacks([EarlyStoppingSpec(monitor="val_dice", patience=2)])[0]
    assert run(callback, state, [0.5, 0.7, 0.7, 0.7], key="val_dice") == 4


def test_early_stopping_restores_the_best_weights(state):
    callback = build_callbacks([EarlyStoppingSpec(patience=1)])[0]
    callback.on_train_begin(state)

    state.epoch, state.logs = 1, {"val_loss": 0.1}
    callback.on_epoch_end(state)
    good = state.model.weight.detach().clone()

    with torch.no_grad():
        state.model.weight.add_(5.0)  # the epoch that made things worse
    state.epoch, state.logs = 2, {"val_loss": 9.9}
    assert callback.on_epoch_end(state)
    callback.on_train_end(state)
    assert torch.allclose(state.model.weight, good)


def test_checkpoint_only_saves_improvements(tmp_path, state):
    path = tmp_path / "best.pt"
    callback = build_callbacks([ModelCheckpointSpec(path=str(path))])[0]
    run(callback, state, [1.0, 0.5, 0.7])
    assert path.exists()
    assert callback.saved_epoch == 2  # not 3, which was worse


def test_checkpoint_can_save_every_epoch(tmp_path, state):
    path = tmp_path / "last.pt"
    callback = build_callbacks([ModelCheckpointSpec(path=str(path), save_best_only=False)])[0]
    run(callback, state, [1.0, 0.5, 0.7])
    assert callback.saved_epoch == 3


def test_reduce_lr_cuts_the_rate_on_a_plateau(state):
    callback = build_callbacks([ReduceLrOnPlateauSpec(factor=0.5, patience=2)])[0]
    run(callback, state, [1.0, 1.0, 1.0])
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_reduce_lr_respects_its_floor(state):
    callback = build_callbacks([ReduceLrOnPlateauSpec(factor=0.01, patience=1, min_lr=0.09)])[0]
    run(callback, state, [1.0, 1.0])
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.09)


def test_csv_logger_writes_a_row_per_epoch(tmp_path, state):
    path = tmp_path / "history.csv"
    callback = build_callbacks([CsvLoggerSpec(path=str(path))])[0]
    run(callback, state, [1.0, 0.5])
    rows = list(csv.DictReader(path.open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[1]["val_loss"] == "0.5"


def test_terminate_on_nan_stops_immediately(state):
    callback = build_callbacks([TerminateOnNaNSpec()])[0]
    assert run(callback, state, [1.0, float("nan")]) == 2
    assert "nan" in state.stop_reason


def test_terminate_on_nan_also_catches_infinity(state):
    callback = build_callbacks([TerminateOnNaNSpec()])[0]
    assert run(callback, state, [float("inf")]) == 1
