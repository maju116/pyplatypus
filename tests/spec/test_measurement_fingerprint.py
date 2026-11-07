"""What a reported figure depends on, beyond the weights and the data.

A weights sidecar recorded `trained_with` and the figures it measured, and **not what the
figures were measured with**. For `dice` that is harmless: it takes `include_background` and
nothing else. For `cldice` it is not - `iterations` is the skeleton's peel count, it moves
the published `fives-unet` number by **0.049**, and it was written nowhere at all, because it
arrived as `ClDice.__init__`'s default through a publisher's own flag. So *is this figure
reproducible* had to be answered by sweeping the parameter until a run landed on the
published digits. `pyplatypus#413` is that day; `#419` is the key that makes it a reading.

`measurement_fingerprint()` is asked of the spec for the reason `weights_fingerprint`'s own
docstring gives one method up: **the answer is task-specific.** Segmentation reports a
configured list of metrics; a detector has no `metrics` field at all and reports one table
whose rows move with three thresholds. A fixed list in `pyplatypus.weights` would have to
grow a branch per task and reach for a field that is not there.

What this file holds is the *class* rather than the four instances of it. Every member of
`AnySpec` is asked, by introspection, so a task added next month either answers or is named
here with a reason - which is the failure this method was written against, since a new task
would otherwise inherit the base class's silence and nothing would say so.
"""

from __future__ import annotations

from typing import get_args

import pytest

from pyplatypus import from_dict
from pyplatypus.spec.models import ModelSpec
from pyplatypus.spec.spec import AnySpec

#: The model class of every task, read off the union rather than listed - `test_task_coverage`
#: does the same and for the same reason: a list maintained by hand is a list that will be
#: wrong, and a task's tag lands last.
MODEL_CLASSES = {
    spec.__name__: get_args(spec.model_fields["models"].annotation)[0]
    for spec in get_args(get_args(AnySpec)[0])
}

#: A task whose figures nothing configurable decides, with the reason. **It is empty**, and
#: that is the state to notice rather than a gap: all four tasks have something, and the
#: entry this list would hold is a task reporting one fixed table with no thresholds and no
#: metric list. An empty exemption list needs the control below, or `assert` over it is an
#: assertion about nothing.
NOTHING_DECIDES_THEIR_FIGURES: dict[str, str] = {}


def test_the_model_classes_were_found_at_all():
    """Or every parametrised case below is a case over an empty mapping."""
    assert len(MODEL_CLASSES) == 4, (
        f"found {sorted(MODEL_CLASSES)} - `AnySpec`'s shape changed, and this file's "
        "introspection has to follow it rather than quietly covering fewer tasks"
    )


def test_the_base_class_answers_EMPTY_so_a_new_task_cannot_inherit_a_wrong_answer():
    """The base is silent on purpose, and the next test is what makes silence loud.

    A base returning something plausible - the metrics, say - would answer for a task that
    has no metrics and read as a record rather than as an omission. Empty plus an
    enumeration is the pair: the method cannot invent, and nothing can stay quiet.
    """
    assert ModelSpec.measurement_fingerprint(object.__new__(ModelSpec)) == {}


@pytest.mark.parametrize("task", sorted(MODEL_CLASSES), ids=sorted(MODEL_CLASSES))
def test_every_task_says_what_its_figures_were_measured_with(task):
    """The enumeration. A task that answers nothing is named above or fails here."""
    model = MODEL_CLASSES[task]
    assert "measurement_fingerprint" in vars(model) or task in NOTHING_DECIDES_THEIR_FIGURES, (
        f"{model.__name__} does not override `measurement_fingerprint`, so it inherits the "
        f"base class's empty answer - which reads as *nothing configurable decides this "
        f"task's figures*. Either that is true and {task} belongs in "
        "NOTHING_DECIDES_THEIR_FIGURES with the reason, or the method is owed"
    )


@pytest.mark.parametrize("task", sorted(MODEL_CLASSES), ids=sorted(MODEL_CLASSES))
def test_what_a_task_names_are_fields_it_actually_has(task):
    """A fingerprint naming a field that no longer exists is a record of nothing.

    Checked against the model's own fields rather than against a list, so a rename fails
    here instead of writing `null` into every sidecar published afterwards. `metrics` is
    the one key that is not a field name - it is the dump of one - and is allowed by name.
    """
    model = MODEL_CLASSES[task]
    if "measurement_fingerprint" not in vars(model):
        pytest.skip(f"{task} is exempt and the test above is what holds that")
    source = inspect_keys(model)
    assert source, f"could not read {model.__name__}'s fingerprint keys"
    for key in source:
        assert key in model.model_fields or key == "metrics", (
            f"{model.__name__}.measurement_fingerprint() records '{key}', which is not one "
            f"of its fields. A renamed field would otherwise write null into every sidecar"
        )


def inspect_keys(model) -> set[str]:
    """The literal keys one class's own `measurement_fingerprint` writes, by AST.

    Its own override only - `super()` is what carries the inherited ones, and those are the
    parent's to justify.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(model.measurement_fingerprint)))
    return {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


# --------------------------------------------------------------------------------------
# And the behaviour, on the case this exists for.


def _segmentation(metrics):
    return from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": "a",
                "validation_path": "b",
                "colormap": [[0, 0, 0], [255, 255, 255]],
            },
            "models": [{"name": "m", "input_shape": [32, 32], "metrics": metrics}],
        },
        check_paths=False,
    ).models[0]


def test_a_metric_parameter_NOBODY_WROTE_is_recorded():
    """The defect's own case, and the whole argument for dumping rather than copying.

    `iterations` is not in this configuration. It is `ClDice`'s default, and it is the value
    that made `#413` a day of measurement - so a fingerprint that recorded the configuration
    as written would record nothing here, which is exactly the state this replaces.
    """
    model = _segmentation([{"name": "cldice"}])
    recorded = model.measurement_fingerprint()["metrics"]

    assert len(recorded) == 1
    assert recorded[0]["name"] == "cldice"
    assert recorded[0]["iterations"] == 5, (
        "the peel count is missing or wrong. It is the parameter whose value moved the "
        "published clDice by 0.049 and appeared in no file"
    )
    # The control that this is a resolved dump and not a passthrough: nothing above wrote
    # `iterations`, and `smooth` is a second default arriving the same way.
    assert "smooth" in recorded[0]


def test_a_value_somebody_DID_write_is_recorded_as_written():
    """The other direction, or the test above passes on a function returning a constant."""
    model = _segmentation([{"name": "cldice", "iterations": 12}])
    assert model.measurement_fingerprint()["metrics"][0]["iterations"] == 12


def test_the_detection_fingerprint_names_the_thresholds_that_move_its_table(detection_config):
    """A detector has no `metrics`, and an empty record would have read as *nothing to say*."""
    model = from_dict(detection_config, check_paths=False).models[0]
    recorded = model.measurement_fingerprint()

    assert set(recorded) == {"score_threshold", "nms_threshold", "operating_point"}, (
        f"detection records {sorted(recorded)}. All three change a published row while the "
        "weights and the data stay the same, which is the test for belonging here"
    )
    assert "metrics" not in recorded, (
        "a detector has no metric list, and writing an empty one would say that nothing "
        "configurable decides its figures"
    )
    # `ignore_threshold` is a training parameter: it shapes the target's ignore mask, so it
    # changes what the weights learnt rather than how they are scored. Asserted so that
    # adding it later is a decision rather than a slip.
    assert "ignore_threshold" in type(model).model_fields
    assert "ignore_threshold" not in recorded
