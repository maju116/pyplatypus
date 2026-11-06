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
