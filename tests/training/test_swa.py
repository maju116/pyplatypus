"""Stochastic weight averaging, and the step that decides whether it works.

An averaged weight tensor *inherits* batch-normalisation statistics from whichever epoch
happened to be last - they were never averaged - so the averaged model is evaluated under
the wrong normalisation unless they are recomputed by a pass over the training data. Left
out, the result is a model that scores far worse than it should with nothing to say why,
which is why most of these tests are about that pass rather than about the average.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyplatypus import build_engine
from pyplatypus.spec.components import Swa as SwaSpec
from pyplatypus.spec.loader import ConfigError, from_dict
from pyplatypus.training.callbacks import Swa, TrainingState, build_callbacks

# --- when averaging begins -----------------------------------------------------------------

@pytest.mark.parametrize(("epochs", "start", "first", "folded"), [
    (10, 0.5, 6, 5),
    (60, 0.75, 46, 15),
    (100, 0.9, 91, 10),
    (60, 1.0, 60, 1),       # the final epoch, never none
    (1, 0.75, 1, 1),        # a one-epoch run still averages something
])
def test_start_is_a_fraction_of_the_run_and_always_folds_at_least_one_epoch(
        epochs, start, first, folded):
    """A fraction rather than an epoch so it survives a change to `epochs`. `start = 1.0`
    averaging nothing would be a field that accepts a value and ignores it."""
    callback = Swa(start=start)
    state = TrainingState(model=nn.Linear(2, 2), optimizer=None, total_epochs=epochs)
    callback.on_train_begin(state)
    assert callback._first == first
    assert sum(1 for e in range(1, epochs + 1) if e >= callback._first) == folded


# --- the batch-norm pass, which is the whole point -----------------------------------------

def _model_with_batch_norm():
    return nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2), nn.ReLU())


def _loader():
    images = torch.randn(8, 1, 6, 6) * 3 + 5          # not unit-normal, so the pass shows
    return DataLoader(TensorDataset(images, torch.zeros(8)), batch_size=4)


def test_the_batch_norm_statistics_are_recomputed_from_the_training_data():
    """Set them to a value no data would produce, then check the pass replaced it."""
    model = _model_with_batch_norm()
    norm = model[1]
    norm.running_mean.fill_(-99.0)
    norm.running_var.fill_(0.5)

    callback = Swa(start=0.5)
    state = TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                          total_epochs=2, train_loader=_loader(), history=[{}])
    callback.on_train_begin(state)
    state.epoch = 2
    callback.on_epoch_end(state)
    callback.on_train_end(state)

    assert norm.running_mean.abs().max() < 90, "the sentinel survived: no pass happened"
    assert torch.isfinite(norm.running_mean).all()
    assert torch.isfinite(norm.running_var).all()


def test_without_a_loader_the_weights_are_still_averaged_and_nothing_pretends_otherwise():
    """`train_loader` is optional on the state, and a caller who did not provide one gets
    the average without the pass rather than an exception - but the sentinel then survives,
    which is the honest observable difference."""
    model = _model_with_batch_norm()
    model[1].running_mean.fill_(-99.0)

    callback = Swa(start=0.5)
    state = TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                          total_epochs=2, history=[{}])
    callback.on_train_begin(state)
    state.epoch = 2
    callback.on_epoch_end(state)
    callback.on_train_end(state)

    assert model[1].running_mean.abs().max() == pytest.approx(99.0)


def test_a_model_without_batch_norm_needs_no_pass_and_asks_for_none():
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU())
    callback = Swa(start=0.5)
    state = TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                          total_epochs=2, train_loader=None, history=[{}])
    callback.on_train_begin(state)
    state.epoch = 2
    callback.on_epoch_end(state)
    callback.on_train_end(state)        # must not raise for want of a loader


# --- what it does and does not touch -------------------------------------------------------

def test_nothing_is_replaced_when_the_run_ended_before_averaging_began():
    """Early stopping can end a run early. Substituting one epoch's weights for "the
    average" would be a lie about what happened, so the final weights are left alone."""
    model = nn.Linear(2, 2)
    before = model.weight.detach().clone()

    callback = Swa(start=0.9)
    state = TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                          total_epochs=100, train_loader=_loader(), history=[{}])
    callback.on_train_begin(state)
    state.epoch = 3                      # nowhere near epoch 91
    callback.on_epoch_end(state)
    callback.on_train_end(state)

    torch.testing.assert_close(model.weight, before)
    assert "swa_epochs_averaged" not in state.history[-1]


def test_the_learning_rate_is_held_once_averaging_begins():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = Swa(start=0.5, learning_rate=0.004)
    state = TrainingState(model=model, optimizer=optimizer, total_epochs=4, history=[{}])
    callback.on_train_begin(state)

    state.epoch = 1
    callback.on_epoch_end(state)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1), "too early to hold it"

    state.epoch = 3
    callback.on_epoch_end(state)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)


# --- through a run -------------------------------------------------------------------------

def test_a_run_reports_how_many_epochs_were_averaged(config, nested_root):
    """In the last history record and not in `state.logs`: by the time `on_train_end` runs
    the per-epoch logs are already written, so a number put there reaches nobody. Found by
    looking for it in the history and not finding it."""
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["models"][0]["epochs"] = 4
    config["models"][0]["callbacks"] = [{"name": "swa", "start": 0.5,
                                         "learning_rate": 1e-3}]
    engine = build_engine(from_dict(config), device="cpu")
    engine.fit()
    record = engine.runs[config["models"][0]["name"]].history.records[-1]
    # round(4 * 0.5) + 1 = 3, so epochs 3 and 4. Written as 3.0 first, from restating the
    # rule instead of reading it - the parametrised test above is where the arithmetic
    # lives and the only place it should be stated.
    assert record["swa_epochs_averaged"] == 2.0


def test_the_final_weights_differ_from_the_same_run_without_it(config, nested_root):
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["models"][0]["epochs"] = 4

    plain = build_engine(from_dict(config), device="cpu")
    plain.fit()

    config["models"][0]["callbacks"] = [{"name": "swa", "start": 0.5,
                                         "learning_rate": 1e-3}]
    averaged = build_engine(from_dict(config), device="cpu")
    averaged.fit()

    name = config["models"][0]["name"]
    one = plain.runs[name].model.state_dict()
    two = averaged.runs[name].model.state_dict()
    floats = [k for k in one if one[k].dtype.is_floating_point]
    assert floats
    assert any(float((one[k].float() - two[k].float()).abs().max()) > 1e-6 for k in floats)


# --- the combination that cancels itself ---------------------------------------------------

def test_swa_with_a_cosine_is_refused_because_the_two_undo_each_other(config, nested_root):
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["models"][0]["callbacks"] = [{"name": "swa"}, {"name": "cosine_annealing"}]
    with pytest.raises(ConfigError, match="undo each other"):
        from_dict(config, check_paths=False)


def test_a_plateau_rescue_is_not_refused(config, nested_root):
    """It lowers the rate on evidence rather than on schedule, and may never fire at all."""
    config["data"]["train_path"] = str(nested_root)
    config["data"]["validation_path"] = str(nested_root)
    config["models"][0]["callbacks"] = [
        {"name": "swa", "learning_rate": 1e-4},
        {"name": "reduce_lr_on_plateau", "monitor": "val_loss"},
    ]
    assert from_dict(config, check_paths=False)


def test_the_spec_builds_the_callback_with_what_it_was_given():
    callback = build_callbacks([SwaSpec(start=0.6, learning_rate=2e-4)])[0]
    assert isinstance(callback, Swa)
    assert callback.start == pytest.approx(0.6)
    assert callback.learning_rate == pytest.approx(2e-4)
