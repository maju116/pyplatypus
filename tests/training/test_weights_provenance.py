"""Which epoch the weights a run ends on came from, asked where that is decided.

`maju116/pyplatypus#403`. `0.8.0a68` made `best_epoch` the epoch a score row's own figures
came from, by recording the provenance in the three places that decide it - early stopping
writes the epoch of the snapshot it restores, the two averagers clear it after loading an
average, and the trainers copy it onto the history - and the twelfth review pass measured
what held that. **Five of the nine sites could be put back with the whole suite passing**:
`swa`'s clear, the classification and detection trainers' copies, and the instance and
classification tables' reads. The two tests written for `#398` each held the one route they
were written on, which is `CLAUDE.md` §4's *guard one level below the change* one layer out -
the guards were at the right level and covered one of three parallel copies.

**So the repair is deduplication rather than five more tests**, which is what `ask_to_stop`,
`finish_training` and `object_labels` all did before it: six of those nine sites were a rule
written out three times. `finish_training` records the provenance itself, in the loop that
decides it, and `engine._provenance_columns` is the one definition of the two columns a score
table carries about the run. A fourth trainer cannot forget the copy, because the argument is
required; a fourth table cannot write the key, because this file says so by AST.

What is left needs a reading rather than a structure, and that is the first test here.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
from torch import nn

from pyplatypus import classification_engine, detection_engine, engine, instance_engine
from pyplatypus.engine import _provenance_columns
from pyplatypus.training.callbacks import (
    Callback,
    EarlyStopping,
    Ema,
    Swa,
    TrainingState,
    finish_training,
)
from pyplatypus.training.trainer import History

PACKAGE = pathlib.Path(engine.__file__).resolve().parent


def _state() -> TrainingState:
    model = nn.Linear(2, 2)
    return TrainingState(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))


class _Replacer(Callback):
    """A weight-replacer that is not one of the three, to rank beside them."""

    replaces_weights = True

    def on_train_end(self, state: TrainingState) -> None:  # pragma: no cover - ranking only
        return


# --- what no structure can hold: which of the replacers has the last word -----------------


def _drive(average, state) -> None:
    """Run the averager far enough that it has something to load.

    Each folds on its own hook - `swa` on an epoch's end, `ema` on a batch's - and both
    refuse to substitute an average they never took, which is the control below. Driven
    rather than reached into, because `_folded == 0` is the condition their `on_train_end`
    actually reads and setting it by hand would test this file's idea of them.
    """
    state.total_epochs = 10
    average.on_train_begin(state)
    for epoch in range(1, 11):
        state.epoch = epoch
        average.on_batch_end(state)
        average.on_epoch_end(state)


@pytest.mark.parametrize(
    ("averager", "name"),
    [
        (lambda: Swa(start_epoch=1, learning_rate=0.05), "swa"),
        (lambda: Ema(decay=0.5), "ema"),
    ],
)
def test_an_average_leaves_no_epoch_even_when_early_stopping_recorded_one(averager, name):
    """The survivor this file exists for, and `swa` is the half that was unheld.

    Both averagers clear the provenance **after** loading, so a run that ended before
    averaging began keeps whatever `restore_best` recorded - and a run that did average
    reports no epoch, because averaged weights are no epoch's. `finish_training` runs the
    replacers in a fixed order, so this asks the question through it rather than by calling
    `on_train_end` by hand: the order is the whole of what makes one of the two answers
    right.
    """
    state = _state()
    history = History()

    early = EarlyStopping(monitor="train_loss", restore_best=True)
    early._best_epoch = 7
    early._best_weights = {
        key: value.detach().clone() for key, value in state.model.state_dict().items()
    }

    average = averager()
    _drive(average, state)
    assert average._folded > 0, "the averager folded nothing, so its clear is unreachable"

    finish_training([average, early], state, history)
    assert history.weights_epoch is None, (
        f"`{name}` loaded an average and the run still reports epoch "
        f"{history.weights_epoch}, which is an epoch those weights are not"
    )


@pytest.mark.parametrize(
    ("averager", "name"),
    [
        # `start_epoch` past the run's own length, and the horizon is what makes that mean
        # something: `Swa.on_train_begin` CLAMPS `_first` to the number of epochs,
        # deliberately, so a start beyond the end folds the last epoch rather than nothing.
        # With ten declared and three run, `_first` is 5 and nothing is folded - a state an
        # unset `total_epochs` cannot produce, which is how the first draft of this control
        # came to average on `start_epoch=99`.
        (lambda: Swa(start_epoch=5, learning_rate=0.05), "swa"),
        (lambda: Ema(decay=0.5), "ema"),
    ],
)
def test_an_averager_that_never_averaged_leaves_the_restored_epoch_standing(averager, name):
    """The control, and without it the two rows above are satisfied by clearing always.

    A run that stopped before averaging began has no average to load, so what
    `restore_best` put back IS an epoch's and the number is the truth. Both averagers are
    asked, because the placement each one needs is *after its own early return* - and that
    is the only part of the placement a reading can see: moving either clear to just before
    the `load_state_dict` changes nothing observable, which is measured rather than assumed.
    """
    state = _state()
    history = History()

    early = EarlyStopping(monitor="train_loss", restore_best=True)
    early._best_epoch = 7
    early._best_weights = {
        key: value.detach().clone() for key, value in state.model.state_dict().items()
    }

    average = averager()
    state.total_epochs = 10
    average.on_train_begin(state)
    for epoch in range(1, 4):
        state.epoch = epoch
        average.on_epoch_end(state)
    assert average._folded == 0, f"the `{name}` control folded something, so it is not one"

    finish_training([average, early], state, history)
    assert history.weights_epoch == 7, (
        f"`{name}` averaged nothing and the run reports no epoch, so what `restore_best` "
        f"put back has been thrown away"
    )


def test_finish_training_is_what_writes_the_provenance_onto_the_history():
    """The one copy, asked directly - three trainers had this line and two were deletable.

    It is `None` before and the state's value after, so a `finish_training` that stopped at
    running the callbacks fails here rather than in whichever engine happened to be tested.
    """
    state = _state()
    state.weights_epoch = 4
    history = History()
    assert history.weights_epoch is None
    finish_training([], state, history)
    assert history.weights_epoch == 4


# --- what AST holds: that nobody has written either copy out again ------------------------

#: Dict keys in the package that say `epochs_run` or `best_epoch` without going through the
#: one definition, each with the reason - so a fourth table gets a row or fails this file.
#:
#: **It is empty, and that is the state `pyplatypus#408` reached.** It held
#: `detection_engine.py: epochs_run` until then, with a reason that turned out to be false -
#: `_provenance_columns`' own docstring has what was wrong with it - and the staleness half
#: below is exactly what failed when the column was finally added. An exemption list that
#: empties is the best outcome such a list can have.
WRITTEN_OUT_ON_PURPOSE: dict[tuple[str, str], str] = {}

#: The positive control for the scan below, and it is needed BECAUSE the list above is empty:
#: with no literal anywhere in the package, `assert found` would be a test of nothing. One
#: line the extractor must find both keys in, which no state of the package can satisfy for
#: it. A string rather than a fixture file, so the control cannot drift away from it.
A_SOURCE_THAT_WRITES_BOTH = (
    'row = {"model": name, "epochs_run": len(history), "best_epoch": history.weights_epoch}'
)


def _keys_in(tree: ast.AST, name: str) -> set[tuple[str, str]]:
    """Every `"epochs_run"` or `"best_epoch"` written as a dict key in one parsed module.

    By AST rather than by grep, so the docstring of `_provenance_columns` - which names both
    columns in prose, as it must - is not a hit. The one definition's own `return` is
    excluded by name, because that is the copy everything else is being compared against.

    Split out from the scan below so the same extractor can be pointed at a string, which is
    what makes "the package holds none" an answer rather than an absence of data.
    """
    inside = {
        node
        for definition in ast.walk(tree)
        if isinstance(definition, ast.FunctionDef) and definition.name == "_provenance_columns"
        for node in ast.walk(definition)
    }
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or node in inside:
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and key.value in {"epochs_run", "best_epoch"}:
                found.add((name, key.value))
    return found


def _literal_column_keys() -> set[tuple[str, str]]:
    """The same extractor over every module in the package."""
    found: set[tuple[str, str]] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        found |= _keys_in(ast.parse(path.read_text()), path.name)
    return found


def test_the_two_run_columns_are_written_in_one_place():
    """The completeness half. A fourth score table that writes its own `best_epoch` is free
    to drift from the three, which is exactly how two of the three drifted far enough to be
    deletable - so it fails here instead."""
    # The control first, and it is the whole reason this reads as a measurement: the package
    # is expected to hold NO literal now, so the extractor has to be shown finding one.
    control = _keys_in(ast.parse(A_SOURCE_THAT_WRITES_BOTH), "<synthetic>")
    assert control == {("<synthetic>", "epochs_run"), ("<synthetic>", "best_epoch")}, (
        f"the extractor cannot find a literal it is pointed straight at: {control}"
    )

    written = _literal_column_keys()
    unexplained = sorted(written - set(WRITTEN_OUT_ON_PURPOSE))
    assert not unexplained, (
        "these write a run column as a literal dict key instead of spreading "
        f"`_provenance_columns`: {', '.join(f'{f}:{k}' for f, k in unexplained)}"
    )

    stale = sorted(set(WRITTEN_OUT_ON_PURPOSE) - written)
    assert not stale, (
        f"WRITTEN_OUT_ON_PURPOSE names keys nothing writes: "
        f"{', '.join(f'{f}:{k}' for f, k in stale)}"
    )


def test_every_score_table_that_reports_an_epoch_goes_through_the_one_definition():
    """The other direction: the three readers are asserted to exist.

    Without this the test above passes on a package where all three tables have quietly
    stopped reporting the column, which is a different defect with the same reading.
    """
    readers = [
        module
        for module in (engine, classification_engine, instance_engine, detection_engine)
        if "_provenance_columns(" in pathlib.Path(module.__file__).read_text()
    ]
    # All FOUR since `pyplatypus#408`: detection was the exception and kept `epochs_run` on
    # its own, so the trainer recorded which epoch the weights came from and the report threw
    # the answer away one line before printing it.
    assert {module.__name__.rsplit(".", 1)[-1] for module in readers} == {
        "engine",
        "classification_engine",
        "instance_engine",
        "detection_engine",
    }


def test_the_one_definition_answers_with_both_columns_and_the_history_s_own_values():
    history = History()
    assert _provenance_columns(history) == {"epochs_run": 0, "best_epoch": None}
    history.records.append({"epoch": 1})
    history.weights_epoch = 1
    assert _provenance_columns(history) == {"epochs_run": 1, "best_epoch": 1}
