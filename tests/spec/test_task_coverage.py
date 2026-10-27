"""Every task this package names is a task every layer can answer about.

`catalogue.py`'s own first paragraph says **a list maintained by hand is a list that will be
wrong**, and that everything in it is read from the pydantic models the schema is generated
from. `available_tasks()` is the exception: three rows written out, because a row carries a
`chooses` tuple and a summary that cannot be derived from anything. That is a legitimate
second copy of one answer - the shape this project keeps a table of - so it is held by a test
instead of by a comment.

What this file is really about is the window `0.8.0a20` was shaped by: a `task` tag lands
**last**, after the branches that read it. Measured on 2026-10-27 by adding one `Task` member
and nothing else, then reverting - four instruments asked about that task, and three lied:

    loader (SPEC_ADAPTER)   refused, with a pydantic tagged-union error
    available_models        ['u_net', 'u_net_plus_plus', 'res_u_net', 'linknet']
    available_losses        11 rows, identical to semantic segmentation's
    available_metrics       5 rows, identical to semantic segmentation's
    available_tasks()       did not list it at all

So the package denied the task existed and described it anyway. `engine_for` was the one
dispatch site already written to refuse, and its docstring already said why. The three
listing functions fell through to semantic segmentation, which is a silent wrong answer
rather than an error, and that is what the second half of this file pins.
"""

from __future__ import annotations

from typing import get_args

import pytest

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.catalogue import (
    available_losses,
    available_metrics,
    available_models,
    available_tasks,
)
from pyplatypus.spec.common import Task
from pyplatypus.spec.spec import AnySpec


def union_tasks() -> set[Task]:
    """The tasks `AnySpec` can actually validate, read off the union.

    Read rather than written out, for the reason `conftest.metric_configs` gives about
    metrics: a test carrying its own list is the same defect one layer up. The assertion is
    the positive control - an extraction that silently finds nothing would make every
    comparison below pass by comparing two empty sets.
    """
    members = get_args(get_args(AnySpec)[0])
    assert members, "read no members off AnySpec; the extraction is the instrument"
    tasks = set()
    for member in members:
        literal = get_args(member.model_fields["task"].annotation)
        assert len(literal) == 1, f"{member.__name__}'s task is not a single-valued Literal"
        tasks.add(literal[0])
    return tasks


def test_every_task_can_be_validated_and_every_validatable_task_is_named():
    """`Task` and `AnySpec` are the same set, in both directions.

    A member in `Task` with no spec in the union is a task a user can write and the loader
    will refuse with a pydantic error about internal validator names. A spec in the union
    whose task is not in `Task` cannot happen while the field is a `Literal[Task.X]`, and the
    comparison is written both ways anyway, because the cheap direction is the one that
    passes when the extraction breaks.
    """
    assert union_tasks() == set(Task)


def test_available_tasks_lists_exactly_what_can_be_validated():
    """The hand-written rows and the union agree.

    This is the test that fails first when a fifth task is added to `Task` and `AnySpec`
    without a row here - which is the direction that matters, since a task that validates
    and is not listed is a feature nobody can find.
    """
    listed = {row["name"] for row in available_tasks()}
    assert listed == {task.value for task in union_tasks()}


def test_every_listed_task_carries_what_a_reader_needs():
    """A row with an empty `chooses` or no description is a row that is there and says
    nothing, which `available_tasks()` exists to avoid being."""
    for row in available_tasks():
        assert row["chooses"], f"{row['name']} chooses nothing"
        assert row["description"].strip(), f"{row['name']} has no description"
        assert "models" in row["chooses"], f"{row['name']} does not choose models"


class _TaskWithNoBranch:
    """Stands in for a `Task` member added before the branches that read it.

    `Task` is a `str, Enum` and cannot be extended at runtime, and the condition under test
    is precisely *a resolved task none of the `is` comparisons match*. So the resolution is
    patched rather than the enum: everything downstream of `_task` then runs for real, which
    is the half that used to fall through.
    """

    value = "a_task_with_no_branch"


@pytest.mark.parametrize(
    "function, what",
    [
        (available_models, "architectures"),
        (available_losses, "losses"),
        (available_metrics, "metrics"),
    ],
)
def test_a_task_with_no_branch_is_refused_rather_than_answered(monkeypatch, function, what):
    """Each listing function refuses, instead of handing back another task's rows.

    Proven able to fail: with the fall-through restored, `available_models` returns the four
    U-Nets here and `available_losses` eleven rows, both of which are semantic
    segmentation's and neither of which is an answer about the task that was asked for.
    """
    monkeypatch.setattr(
        "pyplatypus.spec.catalogue._task", lambda task: _TaskWithNoBranch(), raising=True
    )
    with pytest.raises(PlatypusError) as caught:
        function("a_task_with_no_branch")

    message = str(caught.value)
    assert "a_task_with_no_branch" in message
    assert what in message
    # The message has to say it is a gap rather than reading as a refusal on the merits -
    # `object_detection` genuinely chooses no losses and says so, and these two must not
    # look alike to whoever reads the traceback.
    assert "gap rather than an answer" in message


def test_the_real_tasks_still_answer(monkeypatch):
    """The positive control for the test above: unpatched, every task in the union is
    answered rather than refused.

    Without this, a `_no_branch` raised unconditionally would satisfy every assertion in
    this file - the failure mode §4 of the workbench calls a check calibrated loosely enough
    to admit the failure it was written to prevent, in its opposite form.
    """
    #: Where a listing refuses **on the merits** rather than for want of a branch. Each
    #: entry is a task that genuinely does not choose that component, and the assertion
    #: below is on the message: a refusal that says "gap rather than an answer" is
    #: `_no_branch` and is the thing this file exists to catch.
    on_the_merits = {
        (available_losses, Task.OBJECT_DETECTION),
        (available_metrics, Task.OBJECT_DETECTION),
        (available_models, Task.CLASSIFICATION),
    }

    for task in union_tasks():
        if (available_models, task) in on_the_merits:
            continue
        models = available_models(task)
        assert models, f"{task.value} lists no architectures"

    assert len(available_losses(Task.SEMANTIC_SEGMENTATION)) == 11
    assert len(available_losses(Task.INSTANCE_SEGMENTATION)) == 14
    assert available_metrics(Task.SEMANTIC_SEGMENTATION)
    # Classification answers both of these and with its own sets, which is the half of
    # "the real tasks still answer" that a new task makes worth asserting by name.
    assert {row["name"] for row in available_losses(Task.CLASSIFICATION)} == {"cce", "focal"}
    assert {row["name"] for row in available_metrics(Task.CLASSIFICATION)} == {
        "accuracy",
        "auc",
        "balanced_accuracy",
        "f1",
    }

    for function, task in sorted(on_the_merits, key=lambda pair: (pair[0].__name__, pair[1])):
        with pytest.raises(PlatypusError) as caught:
            function(task)
        assert "gap rather than an answer" not in str(caught.value), (function.__name__, task)


def test_an_unknown_task_name_is_still_refused_by_name():
    """The pre-existing refusal, which the exhaustiveness change routes around rather than
    through: a string that is not a `Task` at all fails in `_task` and lists what is on
    offer, and that message is what a user sees for a typo."""
    with pytest.raises(PlatypusError) as caught:
        available_models("semantic_segmentatoin")

    message = str(caught.value)
    assert "unknown task" in message
    for task in Task:
        assert task.value in message
