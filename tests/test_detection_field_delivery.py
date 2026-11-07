"""The detection fields that decide what `predict` returns, asked whether they arrive.

`maju116/pyplatypus#395`. The eleventh review pass measured five reads in this task as held
by nothing - `score_threshold`, `operating_point`, `ignore_threshold`, `box_loss` and
`min_visibility` - each replaceable by its own default with every test that executes the
line still passing. They are not structural: what they change is which boxes come back and
where precision is read, so a module count cannot see them and the shape that fits is the
spy `#386` used for `nms_threshold`.

Each is asked at **the site that reads it**, which is the positional rule `#395`'s own repair
had to learn twice: a guard on the function underneath is a guard the forward above can be
removed under.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pyplatypus.spec.detection import DetectionModel
from pyplatypus.training.detection_trainer import DetectionTrainer

#: Three grids of two anchors each, in the shape `Yolo3Loss` reads - a grid per scale.
ANCHORS = (
    ((0.25, 0.25), (0.15, 0.30)),
    ((0.10, 0.10), (0.06, 0.12)),
    ((0.04, 0.04), (0.02, 0.05)),
)


class Flat(nn.Module):
    """A detector-shaped stub: three grids of the right width, constant content."""

    def __init__(self, n_class: int = 2, per_grid: int = 2) -> None:
        super().__init__()
        self.width = per_grid * (5 + n_class)
        self.ours = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = x.shape[0]
        return [torch.zeros(batch, self.width, size, size) + self.ours for size in (8, 16, 32)]


def _detection_spec(**overrides):
    base = {
        "name": "d",
        "input_shape": (256, 256),
        "epochs": 1,
        "batch_size": 2,
    }
    return DetectionModel(**{**base, **overrides})


# --- the trainer's two: what the objective is built with ----------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("ignore_threshold", 0.25), ("box_loss", "giou")],
)
def test_the_objective_the_trainer_builds_carries_the_field_the_spec_names(field, value):
    """Read off the built `Yolo3Loss` rather than spied, which is better: the question of
    this whole class is whether the BUILT object reflects the field, and the trainer keeps
    the objective it built.

    A default-valued constant in place of the read is what the pass mutated, so each row
    asks for a NON-default and demands that value rather than merely a value.
    """
    spec = _detection_spec(**{field: value})
    trainer = DetectionTrainer(
        Flat(),
        spec,
        anchors=ANCHORS,
        n_class=2,
        device="cpu",
        callbacks=[],
    )
    got = getattr(trainer.loss_fn, field)
    assert got == value, f"the spec asked for {field}={value!r} and the objective has {got!r}"


def test_the_defaults_are_not_what_those_two_rows_assert():
    """The control that makes the rows above readings rather than tautologies: the values
    they ask for are not the defaults a constant-ified read would supply."""
    default = _detection_spec()
    assert default.ignore_threshold != 0.25
    assert getattr(default.box_loss, "name", default.box_loss) != "giou"


# --- the network the engine builds, which is `#404`'s half ---------------------------------
#
# The twelfth pass replaced `in_channels=model_spec.channels` by a literal and the whole
# suite passed. The read sat inside `train`'s loop over the models, which nothing can enter
# without a full run, so there was no site to ask the question of - `0.8.0a70` extracts
# `_network_for` for exactly that reason and this is what asks it.
#
# A structural reading rather than a fingerprint over the values: both of these change the
# module tree, and a tree is the one answer no OTHER read of the same field can give.

#: field on the model spec -> a value that is not the default
BUILD_ROUTE_FIELDS = {
    "channels": 1,
    "anchors_per_grid": 2,
}


def _reads_in_the_build_route() -> set[str]:
    """Every `model_spec.<name>` inside `_network_for`, by AST rather than by grep."""
    import ast
    import pathlib

    from pyplatypus import detection_engine

    source = pathlib.Path(detection_engine.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_network_for":
            found = {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "model_spec"
            }
            assert found, "`_network_for` was found and reads no `model_spec.<field>` at all"
            return found
    raise AssertionError("no `_network_for` in the detection engine; the scan is misaimed")


@pytest.mark.parametrize("field", sorted(BUILD_ROUTE_FIELDS))
def test_a_field_reaches_the_detector_the_engine_builds(field):
    from pyplatypus.detection_engine import _network_for

    value = BUILD_ROUTE_FIELDS[field]
    assert DetectionModel.model_fields[field].default != value, (
        f"`{field}={value!r}` is the default, so this row cannot see a read replaced by one"
    )
    default = repr(_network_for(_detection_spec(), n_class=2))
    asked = repr(_network_for(_detection_spec(**{field: value}), n_class=2))
    assert default != asked, (
        f"`{field}={value!r}` builds the same detector as the default, so nothing delivers it"
    )


def test_the_detector_s_class_count_comes_from_the_DATA():
    """The third argument of that call, and it is not a model field at all.

    Worth its own row because the alternative - reading it off the model - is what the
    docstring refuses, and a literal there would be invisible to the two rows above.
    """
    from pyplatypus.detection_engine import _network_for

    assert repr(_network_for(_detection_spec(), n_class=2)) != repr(
        _network_for(_detection_spec(), n_class=7)
    )


def test_every_field_the_build_route_reads_is_enumerated():
    """The completeness half, so a fourth argument tomorrow fails this rather than nothing.

    `#404`: the unit is the read site. These two reads are `_network_for`'s and say nothing
    about the same fields read anywhere else, which is why the scan is scoped to it.
    """
    read = _reads_in_the_build_route()
    missing = sorted(read - set(BUILD_ROUTE_FIELDS))
    assert not missing, (
        f"these fields are read by `_network_for` and have no row: {', '.join(missing)}"
    )
    stale = sorted(set(BUILD_ROUTE_FIELDS) - read)
    assert not stale, f"these rows describe reads `_network_for` does not make: {stale}"


#: field -> (the module that consumes it, what consumes it there). `#428`'s other direction
#: for this family: the completeness half below asks *every read in `_network_for` has a
#: row*, so a field that route never reads needs neither a row nor a reason - and a field
#: forwarded nowhere is the defect. Measured 2026-11-07: `DetectionModel.no_object_weight`,
#: planted with a description and no forward, passed all five enumerating files in this
#: suite.
#:
#: `anchors` is the entry worth reading. It is consumed as `model.anchors_as_tuples`, a
#: derived property, so **no `model_spec.anchors` read exists outside the specification
#: layer at all** - which is the third spelling this pair of lists has had to learn, after
#: `self.spec.x` and `getattr(spec, "x")`.
CONSUMED_ELSEWHERE = {
    "name": ("detection_engine.py", "the engine keys its runs by it"),
    "fit": ("detection_engine.py", "whether this is a training run or a scoring one"),
    "weights": ("detection_engine.py", "loaded with their own anchors before anything builds"),
    "architecture": ("detection_engine.py", "chooses the network `_network_for` builds"),
    "anchors": ("detection_engine.py", "through `anchors_as_tuples`, a derived property"),
    "augmentation": ("detection_engine.py", "the loader's pipeline, not the module tree"),
    "batch_size": ("detection_engine.py", "the loader's"),
    "min_visibility": ("detection_engine.py", "drops a box a crop left too little of"),
    "input_shape": ("data/detection.py", "the letterbox's target size"),
    "epochs": ("training/detection_trainer.py", "the loop's length"),
    "callbacks": ("training/detection_trainer.py", "built by `build_callbacks`"),
    "optimizer": ("training/detection_trainer.py", "built by the optimiser table"),
    "box_loss": ("training/detection_trainer.py", "picks the coordinate term"),
    "ignore_threshold": (
        "training/detection_trainer.py",
        (
            "shapes the target's ignore mask, which is why `#419` keeps it OUT of the "
            "measurement fingerprint: it changes what the weights learnt rather than how "
            "they are scored"
        ),
    ),
    "score_threshold": ("detection_engine.py", "the score cut; recorded by `#419`"),
    "nms_threshold": ("detection_engine.py", "suppression; recorded by `#419`"),
    "operating_point": ("detection_engine.py", "the reported row; recorded by `#419`"),
}


def _reads_the_field(text: str, field: str) -> bool:
    """Any spelling a consumer uses, including a derived property over the field."""
    import re

    return bool(
        re.search(rf"\b(?:self\.)?(?:spec|model_spec|model)\.{field}(?:_[a-z0-9_]+)?\b", text)
    )


def test_every_FIELD_is_either_enumerated_or_named_whether_the_route_reads_it_or_not():
    """`#428`: the direction a forgotten forward arrives from.

    The route scan cannot see a field nothing forwards, by construction. So every field of
    the model is accounted for here - a row in `BUILD_ROUTE_FIELDS`, or a named consumer -
    and `test_the_consumer_named_for_a_field_really_reads_it` checks the second kind rather
    than reading it.
    """
    fields = set(DetectionModel.model_fields)
    missing = sorted(fields - set(BUILD_ROUTE_FIELDS) - set(CONSUMED_ELSEWHERE))
    assert not missing, (
        f"these fields of a detection model have no row and no named consumer: "
        f"{', '.join(missing)}. A field forwarded nowhere is what this asks about"
    )

    stale = sorted(set(CONSUMED_ELSEWHERE) - fields)
    assert not stale, f"CONSUMED_ELSEWHERE names fields that do not exist: {stale}"


@pytest.mark.parametrize("field", sorted(CONSUMED_ELSEWHERE))
def test_the_consumer_named_for_a_field_really_reads_it(field):
    """A reason that cannot be followed is decoration - §4's newest category, in test code."""
    import pathlib

    import pyplatypus

    module, why = CONSUMED_ELSEWHERE[field]
    path = pathlib.Path(pyplatypus.__file__).parent / module
    assert path.exists(), f"{field} names {module}, which is not a file"
    assert _reads_the_field(path.read_text(), field), (
        f"{field} is exempt because {why}, in {module} - and {module} does not read it under "
        "any spelling this checker knows. Either the consumer moved or the reason was never "
        "true"
    )
