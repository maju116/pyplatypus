"""Which callbacks see the last epoch, and which weights a run hands back.

Two defects from the reviewer's second pass, and they are the same defect twice: a decision
that nothing stated, taken by where a callback happened to sit in the list.

`pyplatypus#334` - the epoch loop asked `any(... for ...)`, which short-circuits, so a
callback listed after the first one that wanted to stop never saw the epoch the run stopped
on. `pyplatypus#333` - three callbacks replace the model's weights when training ends and
the trainer calls them in list order, so the last one wins and `restore_best` defaults to
`True`.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pyplatypus.data import SegmentationDataset, discover
from pyplatypus.models import build_model
from pyplatypus.spec.components import CsvLogger as CsvLoggerSpec
from pyplatypus.spec.components import DiceMetric
from pyplatypus.spec.components import EarlyStopping as EarlyStoppingSpec
from pyplatypus.spec.components import Ema as EmaSpec
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training import Trainer, make_loader
from pyplatypus.training.callbacks import (
    Callback,
    CsvLogger,
    EarlyStopping,
    Ema,
    TrainingState,
    ask_to_stop,
    build_callbacks,
    finish_training,
)


def model_spec(**overrides):
    base = {
        "name": "m",
        "input_shape": (32, 32),
        "channels": 3,
        "blocks": 2,
        "filters": 4,
        "batch_size": 2,
        "epochs": 4,
        "metrics": [DiceMetric()],
    }
    return SegmentationModel(**{**base, **overrides})


@pytest.fixture
def loaders(nested_root, binary_data):
    samples = discover(nested_root, binary_data).samples

    def build(spec, **kwargs):
        base = SegmentationDataset(samples, spec, binary_data, **kwargs)
        return make_loader(base, batch_size=spec.batch_size)

    return build


class _Stops(Callback):
    """Asks to stop the first time it is asked anything."""

    def on_epoch_end(self, state: TrainingState) -> bool:
        return True


class _Records(Callback):
    """Writes down every epoch it was shown."""

    def __init__(self):
        self.seen: list[int] = []

    def on_epoch_end(self, state: TrainingState) -> bool:
        self.seen.append(state.epoch)
        return False


def test_every_callback_is_asked_even_after_one_has_said_stop():
    """`pyplatypus#334`, at the one definition the three trainers now share.

    The generator form passes the first half of this and fails the second, which is what
    makes the pair worth asserting together rather than only the answer.
    """
    after = _Records()
    state = TrainingState(model=nn.Linear(1, 1), optimizer=None, total_epochs=7, epoch=7)

    assert ask_to_stop([_Stops(), after], state) is True, "the stop was not reported"
    assert after.seen == [7], "a callback after the stopper was skipped"

    # The control: with nobody stopping, the answer is False and the record is the same, so
    # the assertion above is about being ASKED rather than about the return value.
    again = _Records()
    assert ask_to_stop([_Records(), again], state) is False
    assert again.seen == [7]


def test_a_logger_listed_after_a_stopper_still_writes_the_final_epoch(loaders, tmp_path):
    """The run-length CSV was one row short for any stopper written before the logger."""
    path = tmp_path / "run.csv"
    spec = model_spec(epochs=4)
    stopper = EarlyStopping(monitor="train_loss", patience=1, min_delta=1e9)
    trainer = Trainer(
        build_model(spec, n_class=2),
        spec,
        device="cpu",
        callbacks=[stopper, CsvLogger(str(path))],
    )
    history = trainer.fit(loaders(spec), loaders(spec))

    rows = path.read_text().strip().splitlines()[1:]
    assert len(history) == 2, "the stopper did not stop at epoch 2, so this proves nothing"
    assert len(rows) == len(history), f"{len(rows)} rows for {len(history)} epochs"


def test_the_weight_replacers_end_a_run_the_same_way_in_either_order(loaders):
    """`pyplatypus#333`: the pair gave different weights depending on list position.

    `min_delta=1e9` makes epoch 1 the best epoch for ever, so `restore_best`'s snapshot and
    the average are as far apart as this fixture can make them - which is what gives the
    comparison below something to see.
    """

    def run(order):
        torch.manual_seed(1)
        spec = model_spec(epochs=3)
        model = build_model(spec, n_class=2)
        stopper = EarlyStopping(monitor="train_loss", patience=99, min_delta=1e9, restore_best=True)
        ema = Ema(decay=0.5)
        callbacks = [ema, stopper] if order == "ema first" else [stopper, ema]
        trainer = Trainer(model, spec, device="cpu", callbacks=callbacks)
        history = trainer.fit(loaders(spec), loaders(spec))
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return weights, history.records[-1].get("ema_steps")

    first, steps_first = run("ema first")
    second, steps_second = run("stopper first")

    gap = max(float((first[k] - second[k]).abs().max()) for k in first)
    assert gap == pytest.approx(0.0, abs=1e-6), f"list order changed the weights by {gap}"

    # The controls. Both runs must actually have averaged something, or two runs that did
    # nothing would agree perfectly and this test would pass on an `Ema` that never fired.
    assert steps_first and steps_first > 0, "nothing was averaged, so the gap above is vacuous"
    assert steps_second == steps_first


def test_the_averager_is_what_a_run_ends_on_when_restore_best_also_asks(loaders):
    """And the decision is stated rather than inherited from the list.

    `0.8.0a57` chose the average: somebody who configured averaging asked for the averaged
    model, and `restore_best` is a statement about which epoch was best. The test is that
    the pair's answer equals `ema` alone rather than `early_stopping` alone.
    """

    def run(callbacks):
        torch.manual_seed(1)
        spec = model_spec(epochs=3)
        model = build_model(spec, n_class=2)
        Trainer(model, spec, device="cpu", callbacks=callbacks()).fit(loaders(spec), loaders(spec))
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    stopper = lambda: [
        EarlyStopping(monitor="train_loss", patience=99, min_delta=1e9, restore_best=True)
    ]
    averager = lambda: [Ema(decay=0.5)]
    both = lambda: [
        EarlyStopping(monitor="train_loss", patience=99, min_delta=1e9, restore_best=True),
        Ema(decay=0.5),
    ]

    best_only, ema_only, pair = run(stopper), run(averager), run(both)

    to_ema = max(float((pair[k] - ema_only[k]).abs().max()) for k in pair)
    to_best = max(float((pair[k] - best_only[k]).abs().max()) for k in pair)
    assert to_ema == pytest.approx(0.0, abs=1e-6), f"the pair did not end on the average ({to_ema})"
    # The control, and the reason the line above is not vacuous: the two alternatives are
    # genuinely different weights. Without this a run where nothing replaced anything would
    # satisfy both comparisons at once.
    assert to_best > 1e-3, f"the two outcomes are indistinguishable here ({to_best})"


def test_the_epochs_keep_the_order_that_was_written_and_only_the_ending_is_reordered():
    """The two questions are settled in two places, and this is the line between them.

    `build_callbacks` moves `swa` last because of the learning rate it pins during the
    epochs. It does NOT reorder the weight-replacers, because that question is about
    `on_train_end` and has to hold for a caller who builds a `Trainer` with its own
    `callbacks=` list - which is the route the first version of this fix left broken, and
    the test above is what caught it.
    """
    specs = [
        EmaSpec(decay=0.5),
        EarlyStoppingSpec(monitor="train_loss"),
        CsvLoggerSpec(path="x.csv"),
    ]
    built = build_callbacks(specs)
    assert [type(c).__name__ for c in built] == ["Ema", "EarlyStopping", "CsvLogger"]

    # And `finish_training` is where the ending is decided. Subclasses, so the ranking
    # still sees the real types while the bodies only say they ran.
    ended: list[str] = []

    class _Ema(Ema):
        def on_train_end(self, state):
            ended.append("ema")

    class _Early(EarlyStopping):
        def on_train_end(self, state):
            ended.append("early_stopping")

    class _Plain(Callback):
        def on_train_end(self, state):
            ended.append("logger")

    finish_training(
        [_Ema(decay=0.5), _Early(monitor="train_loss", restore_best=True), _Plain()],
        TrainingState(model=nn.Linear(1, 1), optimizer=None),
    )
    assert ended == ["logger", "early_stopping", "ema"], ended

    # The control: `restore_best=False` replaces nothing, so it is not ranked and keeps the
    # position it was written in. Without this the ranking could be sorting on the type
    # alone and would reorder a callback that has no claim on the weights.
    ended.clear()
    finish_training(
        [_Ema(decay=0.5), _Early(monitor="train_loss", restore_best=False), _Plain()],
        TrainingState(model=nn.Linear(1, 1), optimizer=None),
    )
    assert ended == ["early_stopping", "logger", "ema"], ended
