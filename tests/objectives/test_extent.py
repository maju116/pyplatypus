"""`needs_extent` against what the built loss does with extent.

The flag says whether a loss is defined over a *set* of elements, and `ClassificationModel`
refuses the ones that are: one label for a whole image is a sample with no region, so an
overlap has nothing to be a ratio of. A flag is a declaration, and a declaration can be
wrong - this is the measurement that keeps the half of it that matters honest.

**The guard runs in one direction, and the direction is the one that costs something.** A
loss declaring `needs_extent = False` is a loss a classification run will be allowed to
use, so it had better give the same number for one element as for four identical ones:
that is what "it scores each answer on its own" means, and if it were false the loss would
be reading a region that a classification target does not have.

The converse does not hold, and `lovasz` is why. Measured below: it is extent-invariant in
value while its mechanism - a ranking of the per-element errors - is vacuous on one
element. So the flag cannot be derived from this probe, which is exactly why it is declared
and why the exception is pinned here rather than left for somebody to "fix" from the
measurement.
"""

from typing import get_args

import pytest
import torch

from pyplatypus.objectives.losses import build_loss
from pyplatypus.spec.components import LossSpec

# Four samples, three classes, a prediction that is wrong enough to be informative - an
# exact prediction makes several of these agree for reasons that have nothing to do with
# extent, which would be a probe that cannot fail.
LOGITS = torch.tensor(
    [[2.0, 0.5, -1.0], [0.1, 1.5, 0.2], [-0.5, 0.3, 1.2], [1.0, 1.0, 1.0]],
)
TARGET = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
)

#: Measured, both ends, which is what §4 of the workbench asks of a tolerance. The noise is
#: `focal`, which agrees to 1.5e-08 absolute (1e-07 relative, under float32's eps of
#: 1.19e-07) because the two reductions sum a different number of terms; what the flag being
#: wrong moves is `dice`, from 0.1598 to 0.3521, which is 120%. The bound sits five orders
#: of magnitude below one and ten times above the other.
TOLERANCE = 1e-6


def value_at(loss, side: int) -> float:
    """The loss over a target of `side**2` identical elements per sample.

    Identical rather than random, so the per-element answers are the same in both calls and
    the only thing that differs is how many there are. A loss asking for a third tensor is
    given the target again: `boundary` reads it as a distance map and the regression family
    as the mask of where their target is defined, and neither is the quantity under test.
    """
    shaped = [t[:, :, None, None].expand(-1, -1, side, side).contiguous() for t in (LOGITS, TARGET)]
    if loss.needs_distance or loss.needs_mask:
        shaped.append(shaped[1])
    return float(loss(*shaped))


LOSSES = [member() for member in get_args(get_args(LossSpec)[0])]


def test_the_extraction_found_the_losses():
    """Or every parametrised test below passes by running on nothing, which is the shape of
    mistake this project has made inside a guard written to prevent it."""
    assert len(LOSSES) > 10
    assert {spec.needs_extent for spec in LOSSES} == {True, False}, (
        "both values of the flag must occur, or the pairing proves nothing"
    )


@pytest.mark.parametrize(
    "spec", [spec for spec in LOSSES if not spec.needs_extent], ids=lambda spec: spec.name
)
def test_a_loss_that_says_it_needs_no_extent_gives_the_same_number_without_one(spec):
    """The direction that matters: these are the losses a classification label may use."""
    loss = build_loss(spec)
    one, four = value_at(loss, 1), value_at(loss, 2)
    assert one == pytest.approx(four, rel=TOLERANCE), (
        f"{spec.name} declares needs_extent=False but answers {one} for one element and "
        f"{four} for four identical ones"
    )


@pytest.mark.parametrize(
    "spec",
    [spec for spec in LOSSES if spec.needs_extent and spec.name != "lovasz"],
    ids=lambda spec: spec.name,
)
def test_a_loss_defined_over_a_region_reads_the_region(spec):
    """Not the flag's definition - `lovasz` is excluded above and pinned below - but
    evidence for it everywhere else: ten of the eleven move, and by far more than the
    tolerance."""
    loss = build_loss(spec)
    one, four = value_at(loss, 1), value_at(loss, 2)
    assert one != pytest.approx(four, rel=TOLERANCE), (
        f"{spec.name} declares needs_extent=True and is indifferent to it; either the flag "
        f"is wrong or this belongs beside `lovasz`"
    )


def test_lovasz_is_the_measured_exception():
    """The one loss where the probe and the flag disagree, written down rather than
    smoothed over.

    The Lovász extension sorts the per-element errors and weights them by the increments of
    the Jaccard set function. With one element there is nothing to sort and the set function
    has one increment, so what is left is a hinge on that element's error - computable,
    extent-invariant, and no longer an IoU surrogate. The flag answers *is this loss about a
    region*, which is still yes, and this test exists so that the next person to read the
    probe does not conclude the flag is wrong.
    """
    spec = next(spec for spec in LOSSES if spec.name == "lovasz")
    assert spec.needs_extent
    loss = build_loss(spec)
    assert value_at(loss, 1) == pytest.approx(value_at(loss, 2), rel=TOLERANCE)
