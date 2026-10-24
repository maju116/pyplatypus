import csv
import itertools

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
    Swa as SwaSpec,
)
from pyplatypus.spec.components import (
    TerminateOnNaN as TerminateOnNaNSpec,
)
from pyplatypus.training.callbacks import TrainingState, better_is_lower, build_callbacks
from pyplatypus.weights import read_state


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


def test_a_checkpoint_is_written_in_the_format_its_path_names(tmp_path, state):
    """`#256`: the callback wrote a torch archive whatever the path said."""
    path = tmp_path / "best.safetensors"
    callback = build_callbacks([ModelCheckpointSpec(path=str(path))])[0]
    run(callback, state, [1.0, 0.5])

    assert path.read_bytes()[8:9] == b"{"
    assert set(read_state(path)) == set(state.model.state_dict())


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


# ------------------------------------------------------ the learning-rate schedules
# A schedule computes the rate from the optimiser's own initial value and the epoch. The
# two tests that matter are the shape of the curve and the fact that each parameter group
# keeps its own starting rate - the second is what `encoder_learning_rate` depends on, and
# flattening it would be invisible, because the history records one rate: the first group's.
def two_groups(first: float = 0.1, second: float = 0.01):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(
        [{"params": [model.weight], "lr": first}, {"params": [model.bias], "lr": second}]
    )
    state = TrainingState(model=model, optimizer=optimizer)
    state.total_epochs = 10
    return state, optimizer


def rates(callback, state, optimizer, epochs, group=0):
    callback.on_train_begin(state)
    seen = []
    for epoch in range(1, epochs + 1):
        state.epoch, state.logs = epoch, {"val_loss": 1.0}
        callback.on_epoch_end(state)
        seen.append(optimizer.param_groups[group]["lr"])
    return seen


def test_linear_annealing_is_a_straight_line_to_min_lr():
    state, optimizer = two_groups()
    callback = build_callbacks([LinearAnnealingSpec(min_lr=0.0)])[0]

    seen = rates(callback, state, optimizer, 10)
    assert seen[0] == pytest.approx(0.09)
    assert seen[4] == pytest.approx(0.05)
    assert seen[-1] == pytest.approx(0.0, abs=1e-12)
    # A line: every step down is the same size.
    steps = [round(a - b, 12) for a, b in itertools.pairwise(seen)]
    assert len(set(steps)) == 1


def test_a_schedule_past_its_horizon_holds_rather_than_climbing_back():
    state, optimizer = two_groups()
    callback = build_callbacks([LinearAnnealingSpec(min_lr=0.001, epochs=5)])[0]

    seen = rates(callback, state, optimizer, 8)
    assert seen[4] == pytest.approx(0.001)
    assert seen[7] == pytest.approx(0.001)


def test_every_group_descends_from_its_own_rate():
    """`encoder_learning_rate` exists to start the groups apart; a schedule must keep them so."""
    state, optimizer = two_groups(first=0.1, second=0.01)
    callback = build_callbacks([LinearAnnealingSpec(min_lr=0.0)])[0]
    callback.on_train_begin(state)
    state.epoch, state.logs = 5, {"val_loss": 1.0}
    callback.on_epoch_end(state)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.005)


def test_step_decay_drops_on_a_period_and_does_not_drift():
    state, optimizer = two_groups()
    callback = build_callbacks([StepDecaySpec(factor=0.1, every=2)])[0]

    seen = rates(callback, state, optimizer, 6)
    assert seen == pytest.approx([0.1, 0.01, 0.01, 0.001, 0.001, 0.0001])


def test_step_decay_drops_at_the_epochs_a_recipe_names():
    state, optimizer = two_groups()
    callback = build_callbacks([StepDecaySpec(factor=0.1, milestones=[3, 5])])[0]

    seen = rates(callback, state, optimizer, 6)
    assert seen == pytest.approx([0.1, 0.1, 0.01, 0.01, 0.001, 0.001])


def test_step_decay_stops_at_its_floor():
    state, optimizer = two_groups()
    callback = build_callbacks([StepDecaySpec(factor=0.1, every=1, min_lr=0.005)])[0]

    seen = rates(callback, state, optimizer, 4)
    assert seen == pytest.approx([0.01, 0.005, 0.005, 0.005])


def test_swa_is_moved_last_whatever_the_list_said():
    """Two callbacks write the rate in one epoch and the order decided itself before this.

    `swa` pins a constant once averaging starts and a schedule recomputes from the initial
    rate, so whichever runs second is what the epoch ends with. Written in either order,
    the answer has to be the same one.
    """
    for specs in (
        [StepDecaySpec(factor=0.1, every=1), SwaSpec(start=0.01, learning_rate=0.05)],
        [SwaSpec(start=0.01, learning_rate=0.05), StepDecaySpec(factor=0.1, every=1)],
    ):
        state, optimizer = two_groups()
        callbacks = build_callbacks(specs)
        for callback in callbacks:
            callback.on_train_begin(state)
        state.epoch, state.logs = 1, {"val_loss": 1.0}
        for callback in callbacks:
            callback.on_epoch_end(state)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
