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
