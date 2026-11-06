"""The one input that tells `logits > 0` from `logits > 0.5` apart.

`maju116/pyplatypus#404`, the twelfth review pass's F3. Two places in this package harden a
multi-label prediction, and both write `logits > 0` with a comment saying it is
`sigmoid(logits) > 0.5` **exactly** - which it is. Moved to `logits > 0.5`, a different rule
by a wide margin (`p > 0.622`), the whole suite still passed: 2894 tests, nothing red.

The reason is the shape of every fixture rather than their number. 169 executing tests reach
the segmentation site and 8 reach the classification one, and **not one of them predicts a
logit between 0 and 0.5** - a probability between a half and 0.62. Confident logits, which
are what a fixture is written with, give the same answer under both rules, so the two are
indistinguishable to the suite that exists.

So the case is the whole test: a logit of **0.25**, which is positive under the rule this
package documents and negative under the mutation. Each site is asked at the site, and each
row carries its own mirror image - a logit of -0.25, which must NOT read as positive - so
that a rule inverted rather than shifted fails here too.
"""

from __future__ import annotations

import pytest
import torch

from pyplatypus.objectives.classification import (
    MultiLabelAuc,
    MultiLabelBalancedAccuracy,
    MultiLabelMacroF1,
    MultiLabelMetric,
)
from pyplatypus.objectives.metrics import _BUILDERS, build_metric

#: Strictly between 0 and 0.5: `sigmoid(0.25) = 0.562`, so it is a positive prediction under
#: the rule and a negative one under the mutation. The only region where they differ.
BETWEEN = 0.25

#: Every segmentation metric the configuration offers, read off the builders rather than
#: listed - so a sixth one gets these rows without anybody adding it here. Built THROUGH
#: `build_metric`, which is also how `multilabel` arrives in a run: it is a property of the
#: data, and setting the attribute by hand would be a different route from the real one.
SEGMENTATION_SPECS = tuple(sorted(_BUILDERS, key=lambda cls: cls.__name__))

#: What a spec needs before it can be built at all, where its defaults are not enough.
#: `surface_dice`'s `tau` has no default on purpose - a tolerance in millimetres is not a
#: number anybody can pick for somebody else - so it is supplied rather than exempted.
SPEC_ARGUMENTS = {"SurfaceDiceMetric": {"tau": 1.0}}


def _multilabel_metric(spec_class):
    arguments = SPEC_ARGUMENTS.get(spec_class.__name__, {})
    return build_metric(spec_class(**arguments), multilabel=True)


def _one_class_present(logit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """An 8x8 picture whose first channel is two one-pixel lines, one of them the question.

    Lines rather than blocks for `clDice`'s sake, which refuses by name a structure too
    thick for its peels - that refusal is right, so the fixture is what changes. Thin also
    means every metric here has a centreline and a surface, so the five answer one question.

    **Two** lines because one of them is predicted confidently and correctly. That keeps the
    prediction non-empty under either rule, so an absent class scoring a perfect 1 by its
    own smoothing term - which is documented behaviour and not a defect - cannot stand in
    for the answer. The second line carries `logit`, and it is the only thing that moves:
    read as positive the prediction is exactly the target, read as negative it is half of it.
    """
    target = torch.zeros(1, 2, 8, 8)
    target[:, 0, 3, :] = 1.0
    target[:, 0, 6, :] = 1.0
    logits = torch.full((1, 2, 8, 8), -1.0)
    logits[:, 0, 6, :] = 1.0
    logits[:, 0, 3, :] = logit
    return logits, target


@pytest.mark.parametrize("spec_class", SEGMENTATION_SPECS, ids=lambda c: c.__name__)
def test_a_segmentation_metric_calls_a_logit_above_zero_a_positive(spec_class):
    """`SegmentationMetric.forward`, the site - and `0.25` is the input that reads it."""
    metric = _multilabel_metric(spec_class)
    logits, target = _one_class_present(BETWEEN)
    assert float(metric(logits, target)) == pytest.approx(1.0), (
        f"`{metric.name}` scores a logit of {BETWEEN} against a present class below a "
        f"perfect match, so its hardening threshold is not zero"
    )


@pytest.mark.parametrize("spec_class", SEGMENTATION_SPECS, ids=lambda c: c.__name__)
def test_a_segmentation_metric_calls_a_logit_below_zero_a_negative(spec_class):
    """The mirror image, so a rule INVERTED rather than shifted fails here as well.

    Without it every row above is satisfied by `logits > -1`, which is the loose direction
    this project has paid for twice.
    """
    metric = _multilabel_metric(spec_class)
    inverted = float(metric(*_one_class_present(-BETWEEN)))
    assert inverted < 1.0, (
        f"`{metric.name}` scores a logit of {-BETWEEN} against a present class as a perfect "
        f"match, so the sign of the threshold is not what it reads"
    )


def test_a_confident_logit_cannot_tell_the_two_rules_apart():
    """Why the suite was green, asserted rather than described.

    This is the fixture shape the whole package uses, and it reads 1.0 under either rule -
    so the two rows above are the only thing in this file that measures anything.
    """
    metric = _multilabel_metric(next(c for c in SEGMENTATION_SPECS if c.__name__ == "DiceMetric"))
    logits, target = _one_class_present(4.0)
    assert float(metric(logits, target)) == pytest.approx(1.0)
    hard_either_way = (logits > 0) == (logits > 0.5)
    assert bool(hard_either_way.all())
    # And the case the rows above use is exactly where that stops being true.
    between, _ = _one_class_present(BETWEEN)
    assert not bool(((between > 0) == (between > 0.5)).all())


#: The second site: the two multi-label classification metrics that harden. `auc` is not here
#: because it keeps the probabilities instead, which `needs_scores` says and the completeness
#: half below asserts rather than trusting this comment.
HARDENING_CLASSIFICATION = (MultiLabelBalancedAccuracy, MultiLabelMacroF1)


def _half_the_rows_positive(logit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Four images, three classes, the first two images carrying every class.

    Both answers present in every class, because `balanced_accuracy` is a mean over the two
    and refuses by name a class with only one - which is right, and is why the all-positive
    fixture this test started with was the wrong shape rather than a finding.
    """
    target = torch.zeros(4, 3)
    target[:2] = 1.0
    logits = torch.full((4, 3), -logit)
    logits[:2] = logit
    return logits, target


@pytest.mark.parametrize("metric_class", HARDENING_CLASSIFICATION, ids=lambda c: c.name)
def test_a_multilabel_classification_metric_calls_a_logit_above_zero_a_positive(metric_class):
    """`MultiLabelMetric.update`, which builds the two-by-two table from `logits > 0`."""
    metric = metric_class()
    logits, target = _half_the_rows_positive(BETWEEN)
    assert float(metric(logits, target)) == pytest.approx(1.0), (
        f"`{metric_class.name}` reads a logit of {BETWEEN} against a present class as a "
        f"miss, so its threshold is not zero"
    )


@pytest.mark.parametrize("metric_class", HARDENING_CLASSIFICATION, ids=lambda c: c.name)
def test_a_multilabel_classification_metric_calls_a_logit_below_zero_a_negative(metric_class):
    metric = metric_class()
    logits, target = _half_the_rows_positive(-BETWEEN)
    assert float(metric(logits, target)) < 1.0


def test_every_multilabel_metric_either_hardens_here_or_keeps_the_scores():
    """The completeness half, so a fourth multi-label metric gets a row or a reason.

    `needs_scores` is the package's own answer to which of the two a metric is, so this
    reads it rather than restating a list - and it asserts both answers occur, because a
    partition where every member is on one side is not a partition.
    """
    family = [cls for cls in MultiLabelMetric.__subclasses__()]
    assert family, "no multi-label metrics found; the scan is looking in the wrong place"

    hardens = {cls for cls in family if not cls.needs_scores}
    keeps_scores = {cls for cls in family if cls.needs_scores}
    assert hardens and keeps_scores, "every multi-label metric answers `needs_scores` alike"
    assert MultiLabelAuc in keeps_scores

    missing = sorted(cls.__name__ for cls in hardens - set(HARDENING_CLASSIFICATION))
    assert not missing, (
        f"these multi-label metrics harden a prediction and have no row here: {', '.join(missing)}"
    )
