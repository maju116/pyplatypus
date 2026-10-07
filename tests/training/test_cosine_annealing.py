"""The cosine schedule, checked against torch's own.

Written because the specification could not express the run that produced this package's
detection numbers: the measured BCCD run decayed its learning rate and a spec could only
hold it constant.

So the assertion that matters is not "the rate goes down". It is that this is the *same*
schedule as `torch.optim.lr_scheduler.CosineAnnealingLR`, epoch for epoch, because
otherwise comparing a run through the specification with the measured one compares two
different experiments and attributes the difference to the wrong thing - which is exactly
what happened twice before this was settled. See `DETECTION_RECON.md` §12.
"""

from __future__ import annotations

import math
import warnings
from itertools import pairwise

import pytest
import torch
from torch import nn

from pyplatypus.training.callbacks import CosineAnnealing, TrainingState


def one_parameter(lr):
    model = nn.Linear(1, 1)
    return model, torch.optim.SGD(model.parameters(), lr=lr)


def rates_from_callback(initial, epochs, min_lr=0.0, horizon=None):
    model, optimizer = one_parameter(initial)
    callback = CosineAnnealing(min_lr=min_lr, epochs=horizon)
    state = TrainingState(model=model, optimizer=optimizer, total_epochs=epochs)
    callback.on_train_begin(state)

    seen = []
    for epoch in range(1, epochs + 1):
        seen.append(optimizer.param_groups[0]["lr"])  # the rate this epoch trains at
        state.epoch = epoch
        callback.on_epoch_end(state)
    return seen


def rates_from_torch(initial, epochs, min_lr=0.0):
    """The reference. Nothing trains here, so torch's warning about calling `step()`
    before `optimizer.step()` is about a loop that does not exist; silenced so it does not
    print eight times in a clean run."""
    _, optimizer = one_parameter(initial)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
        seen = []
        for _ in range(epochs):
            seen.append(optimizer.param_groups[0]["lr"])
            schedule.step()
    return seen


@pytest.mark.parametrize("epochs", [1, 2, 10, 150])
@pytest.mark.parametrize("min_lr", [0.0, 1e-6])
def test_it_is_the_same_schedule_as_torchs(epochs, min_lr):
    mine = rates_from_callback(1e-4, epochs, min_lr)
    theirs = rates_from_torch(1e-4, epochs, min_lr)
    assert mine == pytest.approx(theirs, rel=1e-12)


def test_the_first_epoch_trains_at_the_rate_that_was_asked_for():
    """Stepping before the first epoch instead of after it would start the run already
    decayed, which is a different experiment and invisible in any log."""
    assert rates_from_callback(1e-4, 10)[0] == pytest.approx(1e-4)


def test_the_last_epoch_trains_just_above_the_floor():
    """A T_max-length cosine reaches `min_lr` *after* the last epoch, which is torch's
    behaviour and worth pinning: the last epoch still trains, at a very small rate.

    The expected value is computed rather than guessed. The first version asserted
    `< initial / 50`, a number invented to look small, and the real figure at ten epochs
    is initial / 40.9 - so the test failed on arithmetic rather than on the code.
    """
    epochs = 10
    rates = rates_from_callback(1e-4, epochs, min_lr=0.0)
    expected = 1e-4 * (1 + math.cos(math.pi * (epochs - 1) / epochs)) / 2
    assert rates[-1] == pytest.approx(expected, rel=1e-12)
    assert 0 < rates[-1] < rates[0] / 40


def test_each_parameter_group_decays_from_its_own_rate():
    """The one that matters with `encoder_learning_rate`: the groups start at different
    rates deliberately, and a schedule computing one rate for all of them would undo that
    at the first epoch while the history showed only `learning_rate`, the first group's."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    optimizer = torch.optim.SGD(
        [
            {"params": model[0].parameters()},
            {"params": model[1].parameters(), "lr": 1e-6},
        ],
        lr=1e-4,
    )

    callback = CosineAnnealing()
    state = TrainingState(model=model, optimizer=optimizer, total_epochs=10)
    callback.on_train_begin(state)
    for epoch in range(1, 6):
        state.epoch = epoch
        callback.on_epoch_end(state)

    fast, slow = (group["lr"] for group in optimizer.param_groups)
    assert fast == pytest.approx(1e-4 * 0.5, rel=1e-9)
    assert slow == pytest.approx(1e-6 * 0.5, rel=1e-9)
    assert fast / slow == pytest.approx(100.0)


def test_a_run_past_the_horizon_holds_at_the_floor():
    """`fit(epochs=...)` can outrun the spec's own count. Without the clamp the cosine
    would climb back up its far side, raising the rate at the end of a long run."""
    rates = rates_from_callback(1e-4, 20, min_lr=1e-7, horizon=10)
    assert rates[-1] == pytest.approx(1e-7)
    assert all(a >= b - 1e-18 for a, b in pairwise(rates)), "never goes back up"


def test_it_refuses_to_guess_the_horizon():
    model, optimizer = one_parameter(1e-4)
    callback = CosineAnnealing()
    state = TrainingState(model=model, optimizer=optimizer, total_epochs=0)
    with pytest.raises(ValueError, match="how many epochs"):
        callback.on_train_begin(state)


# --- the join, not the halves -----------------------------------------------------------


def test_a_spec_that_asks_for_it_gets_it(tmp_path):
    """From the YAML key to the rate the optimizer actually uses.

    The unit tests above prove the schedule; this proves the wiring - the discriminated
    union, the builder, and `total_epochs` reaching the state from the trainer. Each of
    those could be wrong with every test above still passing, which is this project's most
    repeated lesson: green on both halves proves the halves, not the join.
    """
    import numpy as np
    from PIL import Image

    from pyplatypus import Engine, from_dict

    for split in ("train", "valid"):
        for n in range(2):
            sample = tmp_path / split / f"s{n}"
            (sample / "images").mkdir(parents=True)
            (sample / "masks").mkdir(parents=True)
            mask = np.zeros((32, 32, 3), np.uint8)
            mask[8:20, 8:20] = 255
            Image.fromarray(np.full((32, 32, 3), 10 * n, np.uint8)).save(
                sample / "images" / "i.png"
            )
            Image.fromarray(mask).save(sample / "masks" / "m.png")

    spec = from_dict(
        {
            "task": "semantic_segmentation",
            "seed": 1,
            "data": {
                "train_path": str(tmp_path / "train"),
                "validation_path": str(tmp_path / "valid"),
                "colormap": [[0, 0, 0], [255, 255, 255]],
            },
            "models": [
                {
                    "name": "u",
                    "input_shape": [32, 32],
                    "blocks": 2,
                    "filters": 4,
                    "epochs": 4,
                    "batch_size": 2,
                    "optimizer": {"name": "adam", "learning_rate": 1e-3},
                    "callbacks": [{"name": "cosine_annealing"}],
                }
            ],
        }
    )
    history = Engine(spec, device="cpu", check_masks=False).fit()["u"]
    rates = [record["learning_rate"] for record in history.records]

    assert rates[0] == pytest.approx(1e-3), "the first epoch trains at the stated rate"
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0] / 2
    # The same values the callback produces on its own, so the wiring did not quietly
    # substitute some other horizon.
    assert rates == pytest.approx(rates_from_callback(1e-3, 4))
