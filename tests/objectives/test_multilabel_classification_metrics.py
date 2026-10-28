"""Metrics over a label that is a set, where a confusion matrix is not the state.

The tables in here are built by hand rather than drawn, because every claim this family
makes is about an *ordering* between two models and a random draw can only show it on
average. `pyplatypus#307`.
"""

import pytest
import torch

from pyplatypus.objectives.classification import (
    _MULTILABEL_BUILDERS,
    ClassificationMetric,
    MultiLabelAuc,
    MultiLabelBalancedAccuracy,
    MultiLabelMacroF1,
    MultiLabelMetric,
    _auc,
    build_classification_metric,
    build_classification_metrics,
)
from pyplatypus.spec.catalogue import _members
from pyplatypus.spec.components import (
    AccuracyMetric,
    AucMetric,
    BalancedAccuracyMetric,
    ClassificationMetricSpec,
    F1Metric,
)

LARGE = 8.0  # a logit far enough either side of 0 that `logits > 0` is the decision asked for


def table(per_class: list[tuple[int, int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and a multi-hot target realising `(tp, fp, fn, tn)` per class exactly.

    One image per row, so the classes are laid side by side with their own counts and
    nothing is correlated between them - which is what makes a hand-built expectation
    arithmetic rather than a simulation.
    """
    height = max(sum(counts) for counts in per_class)
    logits = torch.full((height, len(per_class)), -LARGE)
    target = torch.zeros((height, len(per_class)))
    for klass, (tp, fp, fn, tn) in enumerate(per_class):
        assert tp + fp + fn + tn == height, "every class needs the same number of images"
        row = 0
        for count, said_yes, is_yes in ((tp, 1, 1), (fp, 1, 0), (fn, 0, 1), (tn, 0, 0)):
            for _ in range(count):
                logits[row, klass] = LARGE if said_yes else -LARGE
                target[row, klass] = float(is_yes)
                row += 1
    return logits, target


def test_the_three_metrics_are_the_arithmetic_of_their_own_tables():
    # One class, so there is no averaging to hide a mistake in the formula.
    logits, target = table([(6, 20, 4, 70)])

    # sensitivity 6/10, specificity 70/90
    assert MultiLabelBalancedAccuracy()(logits, target).item() == pytest.approx((0.6 + 70 / 90) / 2)
    # 2*6 / (2*6 + 20 + 4)
    assert MultiLabelMacroF1()(logits, target).item() == pytest.approx(12 / 36)


def test_the_average_is_over_classes_and_each_carries_its_own_weight():
    # A common class the model handles and a rare one it does not. Averaged over classes,
    # the rare one counts as much - which is the whole reason this is not accuracy.
    logits, target = table([(40, 0, 0, 60), (0, 0, 10, 90)])

    expected = ((1.0 + 1.0) / 2 + (0.0 + 1.0) / 2) / 2
    assert MultiLabelBalancedAccuracy()(logits, target).item() == pytest.approx(expected)
    assert MultiLabelMacroF1()(logits, target).item() == pytest.approx((1.0 + 0.0) / 2)


def test_each_class_decides_for_itself_rather_than_competing():
    """The one tensor-level difference from the single-label family.

    An image with two findings is not a row of a confusion matrix, and an `argmax` over
    these logits would name one of the two and drop the other. Both classes are positive
    in every image here and the model says so, so the only way to score 1.0 is to read the
    classes independently - a competing reading would have to call one of them wrong.
    """
    logits = torch.full((20, 2), LARGE)
    target = torch.ones((20, 2))

    assert MultiLabelMacroF1()(logits, target).item() == pytest.approx(1.0)
    # And `balanced_accuracy` refuses this pass rather than scoring it, because there is no
    # negative of either class to measure a specificity against.
    with pytest.raises(ValueError, match="no negative"):
        MultiLabelBalancedAccuracy()(logits, target)


def test_a_pass_added_in_batches_is_the_pass_scored_in_one_call():
    logits, target = table([(6, 20, 4, 70), (12, 8, 8, 72)])

    for metric_class in (MultiLabelBalancedAccuracy, MultiLabelMacroF1, MultiLabelAuc):
        one_shot = metric_class()(logits, target).item()
        batched = metric_class()
        batched.reset()
        for start in range(0, logits.shape[0], 7):
            batched.update(logits[start : start + 7], target[start : start + 7])
        assert batched.compute() == pytest.approx(one_shot, abs=0.0), metric_class.__name__


def test_the_auc_is_the_same_read_off_the_probability_or_the_logit():
    """The contrast with the single-label family, which is 0.0866 apart over two classes.

    A sigmoid is monotone in its own logit and in nothing else, so the ordering within a
    class cannot move. Asserted to zero rather than approximately, because anything else
    would mean the ranking had changed.
    """
    torch.manual_seed(7)
    logits = torch.randn(300, 3) * 2
    target = (torch.rand(300, 3) < 0.3).float()

    by_probability = MultiLabelAuc()(logits, target).item()
    by_logit = sum(_auc(logits[:, k].double(), target[:, k].long()) for k in range(3)) / 3
    assert by_probability - by_logit == 0.0

    # And what is *stored* is the probability, which no value above can distinguish - a
    # monotone transform of a ranking is the same ranking, so the mutation that stores the
    # logit instead changes nothing any of these numbers can see. The claim is about the
    # state rather than the score, so it is asserted there: a calibration curve is drawn
    # from what is kept, not from the AUC.
    metric = MultiLabelAuc()
    metric.reset()
    metric.update(logits, target)
    kept = torch.cat(metric._scores)
    assert torch.equal(kept, logits.sigmoid())
    assert float(kept.min()) >= 0.0 and float(kept.max()) <= 1.0


def test_a_class_the_pass_holds_one_side_of_is_refused_by_name():
    absent = table([(6, 20, 4, 70), (0, 0, 0, 100)])
    with pytest.raises(ValueError, match=r"\[1\] have no positive"):
        MultiLabelBalancedAccuracy()(*absent)
    with pytest.raises(ValueError, match=r"\[1\] appear in no example"):
        MultiLabelMacroF1()(*absent)

    # F1 asks for less than the other two, which is the one asymmetry in this family: a
    # finding present in every image has a perfectly good F1 and no specificity.
    everywhere = table([(100, 0, 0, 0)])
    assert MultiLabelMacroF1()(*everywhere).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no negative"):
        MultiLabelBalancedAccuracy()(*everywhere)


def test_a_single_label_target_is_refused_rather_than_read_as_one_row():
    logits = torch.randn(10, 3)
    with pytest.raises(ValueError, match="single-label shape"):
        MultiLabelMacroF1()(logits, torch.zeros(10, dtype=torch.long))
    with pytest.raises(ValueError, match="no spatial axis"):
        MultiLabelMacroF1()(torch.randn(10, 3, 4), torch.zeros(10, 3, 4))


def test_a_value_before_any_batch_is_refused_rather_than_zero():
    for metric_class in (MultiLabelBalancedAccuracy, MultiLabelMacroF1, MultiLabelAuc):
        with pytest.raises(ValueError, match="before any batch"):
            metric_class().compute()


def test_the_collapse_everybody_ships_is_below_every_model_with_a_signal():
    """The measurement that excluded `accuracy`, as an ordering rather than a number.

    Two models over four classes of 100 images at a 10% prevalence. One answers no to
    everything; the other finds 6 of the 10 and pays 20 false positives for them. Both
    readings of accuracy prefer the first.
    """
    collapse = table([(0, 0, 10, 90)] * 4)
    signal = table([(6, 20, 4, 70)] * 4)

    for metric_class in (MultiLabelBalancedAccuracy, MultiLabelMacroF1, MultiLabelAuc):
        worse = metric_class()(*collapse).item()
        better = metric_class()(*signal).item()
        assert better > worse, metric_class.__name__

    # And accuracy, computed here because it is not offered: this is what the refusal is
    # about, and a test that only asserted the three above would not say why they are the
    # three. Per class, and over the whole label.
    def per_class_accuracy(logits, target):
        right = ((logits > 0) == (target > 0)).double()
        return right.mean(dim=0).mean().item()

    def exact_match(logits, target):
        return ((logits > 0) == (target > 0)).all(dim=1).double().mean().item()

    assert per_class_accuracy(*collapse) > per_class_accuracy(*signal)
    assert exact_match(*collapse) > exact_match(*signal)
    assert per_class_accuracy(*collapse) == pytest.approx(0.90)
    assert per_class_accuracy(*signal) == pytest.approx(0.76)


def test_the_builder_returns_a_different_object_rather_than_setting_a_flag():
    assert isinstance(build_classification_metric(AucMetric(), multilabel=True), MultiLabelAuc)
    single = build_classification_metric(AucMetric())
    assert isinstance(single, ClassificationMetric)
    assert not isinstance(single, MultiLabelMetric)

    built = build_classification_metrics(
        [BalancedAccuracyMetric(), F1Metric(), AucMetric()], multilabel=True
    )
    # Keyed by the same names either way, which is what lets a callback watch `val_auc`
    # without knowing how the classes were named.
    assert sorted(built) == ["auc", "balanced_accuracy", "f1"]
    assert all(isinstance(m, tuple(_MULTILABEL_BUILDERS.values())) for m in built.values())


def test_accuracy_is_refused_by_the_builder_too_and_the_message_says_why():
    with pytest.raises(KeyError, match="reward a collapse"):
        build_classification_metric(AccuracyMetric(), multilabel=True)


def test_the_specifications_flag_is_the_set_of_metrics_that_have_an_implementation():
    """§3F's pair: the flag is what a configuration is refused on and this is what exists.

    Both directions, because one of them is the failure that actually happens - a metric
    added to the family and not to the mapping reads as admissible and then raises while a
    run is being built. And both values have to occur, or a pairing over a family that
    answers alike proves nothing.
    """
    members = list(_members(ClassificationMetricSpec))
    assert len(members) >= 4

    flagged = {member for member in members if member.supports_multilabel}
    implemented = {member for member in members if member in _MULTILABEL_BUILDERS}
    assert flagged == implemented

    assert flagged, "nothing says it supports a multi-label label"
    assert set(members) - flagged, "every metric claims to - the flag distinguishes nothing"
