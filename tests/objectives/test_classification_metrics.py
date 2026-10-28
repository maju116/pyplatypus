"""A label's metrics, and the protocol that is not the mask metrics' protocol.

The claim these tests exist for is that a classifier's figures are accumulated over a pass
and computed once, rather than averaged over batches the way a mask's are. Every test below
is either that claim, the measurement that justifies a default, or a control for an
instrument used by the others.
"""

from __future__ import annotations

import math
from typing import get_args

import pytest
import torch

from pyplatypus.objectives import functional as f
from pyplatypus.objectives.classification import (
    Accuracy,
    Auc,
    BalancedAccuracy,
    ClassificationMetric,
    MacroF1,
    _auc,
    build_classification_metric,
    build_classification_metrics,
)
from pyplatypus.spec.components import AccuracyMetric, ClassificationMetricSpec

MEMBERS = get_args(get_args(ClassificationMetricSpec)[0])
BUILT = (Accuracy, BalancedAccuracy, MacroF1, Auc)


def calibrated(n: int, prevalence: float, discrimination: float, seed: int = 7):
    """Logits whose softmax **is** the probability the labels were drawn from.

    The model is then perfectly calibrated by construction and only its discrimination
    varies, which is what makes the comparison below about the metrics rather than about
    somebody's choice of threshold. `b` is bisected so the prevalence is the same at every
    discrimination - without that, a sharper model would also be a model of a different
    problem.
    """
    torch.manual_seed(seed)
    x = torch.randn(n, dtype=torch.float64)
    low, high = -40.0, 10.0
    for _ in range(80):
        middle = (low + high) / 2
        if torch.sigmoid(discrimination * x + middle).mean() > prevalence:
            high = middle
        else:
            low = middle
    probability = torch.sigmoid(discrimination * x + middle)
    labels = (torch.rand(n, dtype=torch.float64) < probability).long()
    odds = torch.log(probability / (1 - probability))
    logits = torch.stack([torch.zeros(n, dtype=torch.float64), odds], dim=1)
    assert torch.allclose(logits.softmax(dim=1)[:, 1], probability)
    return logits, labels


def accumulated(metric: ClassificationMetric, logits, labels, batch: int) -> float:
    metric.reset()
    for start in range(0, logits.shape[0], batch):
        metric.update(logits[start : start + batch], labels[start : start + batch])
    return metric.compute()


# --------------------------------------------------------------------------- the protocol


@pytest.mark.parametrize("built", BUILT)
@pytest.mark.parametrize("batch", [8, 97, 1024])
def test_the_value_does_not_depend_on_how_the_pass_was_cut_into_batches(built, batch):
    """The whole point of accumulating rather than averaging, and it has to be exact.

    Not `approx`: a confusion matrix is integer counts and a ranking is a permutation, so
    there is no arithmetic here whose answer may drift with the order of the additions. A
    tolerance would also hide the defect this was written after - `forward` wrapped its
    answer in a default-dtype tensor, which rounded it to float32 and put 2.5e-08 between
    the two routes.
    """
    logits, labels = calibrated(4000, 0.03, 2.0)
    assert accumulated(built(), logits, labels, batch) == built()(logits, labels).item()


@pytest.mark.parametrize("member", MEMBERS)
def test_every_metric_the_configuration_offers_can_be_built_and_names_itself(member):
    """Read off the union, so a fifth metric is covered by arriving rather than by being
    added here - which is the defect `0.8.0a27` shipped, one layer up."""
    built = build_classification_metric(member())
    assert built.name == member.model_fields["name"].default
    logits, labels = calibrated(400, 0.2, 2.0)
    value = built(logits, labels).item()
    assert 0.0 <= value <= 1.0


def test_the_mean_over_batches_is_right_for_accuracy_and_wrong_for_the_other_three():
    """The measurement the protocol rests on, with accuracy as its own positive control.

    If this reported a difference for all four, the instrument would be suspect; if it
    reported one for none, the protocol would be unnecessary. Accuracy agreeing exactly is
    what makes the other three's disagreement a property of those metrics rather than of
    how this test cuts the pass up.
    """
    logits, labels = calibrated(4000, 0.03, 2.0)
    moved = {}
    for built in BUILT:
        pooled = built()(logits, labels).item()
        metric, total, counted = built(), 0.0, 0
        for start in range(0, logits.shape[0], 32):
            chunk, truth = logits[start : start + 32], labels[start : start + 32]
            try:
                total += metric(chunk, truth).item() * chunk.shape[0]
            except ValueError:
                continue  # the batch holds none of a class, which is most of them
            counted += chunk.shape[0]
        moved[metric.name] = abs(total / counted - pooled)

    assert moved["accuracy"] == 0.0
    for name in ("balanced_accuracy", "f1", "auc"):
        assert moved[name] > 1e-3, (name, moved)


def test_a_batch_at_a_low_prevalence_usually_holds_none_of_the_rare_class():
    """Why the three disagree, stated as a count rather than as an explanation.

    This is the mechanism behind the test above, and it is worth its own assertion because
    it is the thing that would change if somebody decided the metrics could be averaged
    after all: the objection is not precision, it is that most batches have no value.
    """
    _, labels = calibrated(4000, 0.03, 2.0)
    empty = sum(1 for start in range(0, 4000, 8) if labels[start : start + 8].sum() == 0)
    assert empty / (4000 // 8) > 0.7, empty


# ------------------------------------------------------------------------- the default


def test_the_auc_is_the_only_one_that_orders_models_by_discrimination():
    """Why `auc` is the default and `accuracy` is not - and why `balanced_accuracy`, the
    obvious fix, is not either.

    A calibrated model at a low prevalence never crosses the threshold, so every metric
    read off the hard prediction sees the same all-negative answer whether the model learnt
    anything or not. Being threshold-free is the property that separates them, and that is
    a different claim from accuracy being unbalanced.
    """
    doses = (0.0, 0.5, 1.0, 2.0, 4.0)
    table = {}
    for built in BUILT:
        row = []
        for dose in doses:
            logits, labels = calibrated(40000, 0.03, dose)
            row.append(built()(logits, labels).item())
        table[built().name] = row

    def monotone(row):
        return all(row[i] <= row[i + 1] + 1e-12 for i in range(len(row) - 1))

    assert monotone(table["auc"]), table["auc"]
    assert not monotone(table["accuracy"]), table["accuracy"]
    # and the span, which is what makes the ordering worth having rather than merely right
    assert max(table["auc"]) - min(table["auc"]) > 0.4
    assert max(table["accuracy"]) - min(table["accuracy"]) < 0.02
    # the signal at the second dose is real, and only the AUC can see it
    assert table["auc"][1] - table["auc"][0] > 0.1
    for blind in ("accuracy", "balanced_accuracy", "f1"):
        assert abs(table[blind][1] - table[blind][0]) < 0.01, (blind, table[blind])


# ------------------------------------------------------------- one formula, not two copies


def test_macro_f1_is_dice_on_a_label_and_is_computed_through_it():
    """A one-hot label vector is a mask of one pixel, so F1 is what Dice is called when the
    pixel is an image. Asserted on the formula rather than trusted from the docstring,
    because the alternative - a second copy of one formula - is what §3F of the workbench's
    notes is a list of."""
    torch.manual_seed(3)
    n, classes = 500, 3
    labels = torch.randint(0, classes, (n,))
    labels[: int(0.8 * n)] = 0
    logits = torch.randn(n, classes, dtype=torch.float64)
    logits += torch.nn.functional.one_hot(labels, classes).double() * 1.5

    hard = torch.nn.functional.one_hot(logits.argmax(dim=1), classes).double()
    onehot = torch.nn.functional.one_hot(labels, classes).double()
    true_positive = (hard * onehot).sum(dim=0)
    false_positive = (hard * (1 - onehot)).sum(dim=0)
    false_negative = ((1 - hard) * onehot).sum(dim=0)
    by_hand = (2 * true_positive / (2 * true_positive + false_positive + false_negative)).mean()

    assert MacroF1()(logits, labels).item() == pytest.approx(by_hand.item(), abs=1e-12)
    assert f.dice_from_overlaps(
        true_positive, false_positive, false_negative, smooth=0.0
    ).mean().item() == pytest.approx(by_hand.item(), abs=1e-12)


def test_micro_averaging_would_be_accuracy_which_is_why_it_is_not_offered():
    """The absence of an `average` field, held to the reason for it."""
    torch.manual_seed(3)
    n, classes = 500, 3
    labels = torch.randint(0, classes, (n,))
    logits = torch.randn(n, classes, dtype=torch.float64)
    logits += torch.nn.functional.one_hot(labels, classes).double() * 1.5

    hard = logits.argmax(dim=1)
    micro_true_positive = (hard == labels).sum().item()
    micro = 2 * micro_true_positive / (2 * micro_true_positive + 0 + 0)
    # every example contributes exactly one prediction and one truth, so a false positive
    # for one class is the false negative of another and the two cancel in the micro sum
    assert micro == pytest.approx(1.0)
    assert Accuracy()(logits, labels).item() == pytest.approx(micro_true_positive / n, abs=1e-12)
    assert not any("average" in member.model_fields for member in MEMBERS)


# ------------------------------------------------------------------------ the AUC itself


def pairwise_auc(scores, positive):
    """Every pair counted. The control the rank form cannot pass by accident, and the thing
    the rank form exists to avoid: this is 400 million comparisons on a 40000-image pass."""
    of_the_class, of_the_rest = scores[positive == 1], scores[positive == 0]
    difference = of_the_class[:, None] - of_the_rest[None, :]
    wins = (difference > 0).double().sum() + 0.5 * (difference == 0).double().sum()
    return (wins / (of_the_class.numel() * of_the_rest.numel())).item()


def test_the_ranks_agree_with_counting_every_pair_including_on_nothing_but_ties():
    """Ties are the half of this that a sort gets wrong silently: a model reporting the same
    probability twice must score those two examples as a coin flip rather than in whatever
    order `argsort` happened to put them."""
    scores = torch.tensor([0.5, 0.5, 0.5, 0.1, 0.9, 0.9], dtype=torch.float64)
    positive = torch.tensor([1, 0, 1, 0, 1, 0])
    assert _auc(scores, positive) == pytest.approx(pairwise_auc(scores, positive), abs=1e-12)

    torch.manual_seed(5)
    heavy = torch.rand(600, dtype=torch.float64).round(decimals=1)
    which = (torch.rand(600) < 0.3).long()
    assert heavy.unique().numel() < 20, "the point of this draw is that it is mostly ties"
    assert _auc(heavy, which) == pytest.approx(pairwise_auc(heavy, which), abs=1e-12)


def test_two_classes_need_no_special_case():
    """Macro one-vs-rest over two classes is the binary AUC exactly, because one score is
    one minus the other - so the general definition covers the common case and there is no
    branch for it to go out of step with."""
    torch.manual_seed(5)
    n = 5000
    labels = (torch.rand(n) < 0.2).long()
    logits = torch.randn(n, 2, dtype=torch.float64)
    logits[:, 1] += 1.5 * labels
    probability = logits.softmax(dim=1)

    one = _auc(probability[:, 1], labels)
    zero = _auc(probability[:, 0], 1 - labels)
    assert one == pytest.approx(zero, abs=1e-12)
    assert Auc()(logits, labels).item() == pytest.approx(one, abs=1e-12)


def test_the_auc_reads_the_probability_rather_than_the_logit():
    """A choice with a measured size rather than a formality: `softmax` is not monotone in
    one logit on its own, so ranking by the logit is a different number - and it is the
    number that ignores how sure the model was about everything else."""
    torch.manual_seed(13)
    n, classes = 4000, 2
    labels = torch.randint(0, classes, (n,))
    logits = torch.randn(n, classes, dtype=torch.float64)
    logits += torch.nn.functional.one_hot(labels, classes).double() * 1.2

    probability = logits.softmax(dim=1)
    by_probability = (
        sum(_auc(probability[:, k], (labels == k).long()) for k in range(classes)) / classes
    )
    by_logit = sum(_auc(logits[:, k], (labels == k).long()) for k in range(classes)) / classes

    assert Auc()(logits, labels).item() == pytest.approx(by_probability, abs=1e-12)
    assert abs(by_probability - by_logit) > 0.01, (by_probability, by_logit)


# ------------------------------------------------------------- what is retained, and what
# ------------------------------------------------------------- is refused


def test_only_the_auc_keeps_every_score_and_both_answers_occur():
    """A pairing proves nothing where every member answers alike, which is the second half
    of §3F's whole-mask test and the reason it is asserted here too."""
    flags = {built().name: built.needs_scores for built in BUILT}
    assert set(flags.values()) == {True, False}, flags
    assert flags["auc"] is True
    assert [name for name, needs in flags.items() if needs] == ["auc"]

    logits, labels = calibrated(200, 0.3, 2.0)
    counting, ranking = Accuracy(), Auc()
    counting.update(logits, labels)
    ranking.update(logits, labels)
    # four numbers against four hundred, and the first does not grow with the pass
    assert counting._confusion.numel() == 4
    assert sum(chunk.numel() for chunk in ranking._scores) == 400
    assert counting._scores == []
    assert ranking._confusion is None


@pytest.mark.parametrize("built", BUILT)
def test_a_value_is_refused_before_anything_was_added(built):
    """Rather than a zero, which would read as a model getting everything wrong."""
    with pytest.raises(ValueError, match="before any batch"):
        built().compute()


@pytest.mark.parametrize("built", [BalancedAccuracy, MacroF1])
def test_a_class_the_pass_never_contained_is_refused_by_number(built):
    """Skipping it is the alternative and it is worse: a mean over the classes that happened
    to appear is a different quantity under the same name."""
    logits = torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\[1\]"):
        built()(logits, torch.tensor([0, 0]))


def test_accuracy_is_the_one_that_survives_a_missing_class():
    """It needs no per-class denominator, and saying so is what the refusal above points at."""
    logits = torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    assert Accuracy()(logits, torch.tensor([0, 0])).item() == 1.0


def test_an_auc_needs_something_to_rank_against():
    with pytest.raises(ValueError, match="an example of the class"):
        Auc()(torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float64), torch.tensor([0, 0]))


def test_a_mask_shaped_tensor_is_refused_rather_than_pooled():
    """A classifier pools before its head, so logits with a spatial axis mean the caller
    handed a metric the wrong model's output."""
    with pytest.raises(ValueError, match="examples, classes"):
        Accuracy()(torch.zeros(2, 2, 4, 4), torch.zeros(2, dtype=torch.long))


def test_either_shape_of_target_is_accepted_and_they_agree():
    """Indices from a reader that read a directory name, one-hot from anything that built it
    the way a mask is built."""
    logits, labels = calibrated(300, 0.3, 2.0)
    onehot = torch.nn.functional.one_hot(labels, 2).double()
    for built in BUILT:
        assert built()(logits, labels).item() == built()(logits, onehot).item()


# ---------------------------------------------------------------- the guard in `overlaps`


def test_overlaps_refuses_a_tensor_with_no_spatial_axis():
    """`sum(dim=())` sums over every axis rather than none, which is the whole reason this
    guard exists: without it a `(batch, class)` tensor returns one number for the batch and
    it looks exactly like a count."""
    assert torch.ones(4, 3).sum(f.spatial_dims(torch.ones(4, 3))).item() == 12.0
    with pytest.raises(ValueError, match="at least one spatial axis"):
        f.overlaps(torch.ones(4, 3), torch.ones(4, 3))
    # and it still answers for the shape every caller in this package actually passes
    counts = f.overlaps(torch.ones(2, 3, 4, 4), torch.ones(2, 3, 4, 4))
    assert [tuple(part.shape) for part in counts] == [(2, 3)] * 3


def test_the_builders_key_by_name():
    built = build_classification_metrics([member() for member in MEMBERS])
    assert sorted(built) == sorted(member.model_fields["name"].default for member in MEMBERS)
    assert all(isinstance(value, ClassificationMetric) for value in built.values())
    assert not math.isnan(built["accuracy"](*calibrated(100, 0.3, 2.0)).item())


def test_the_confusion_matrix_is_moved_off_the_device_before_it_becomes_float64():
    """The same defect one family over, and this one **shipped** in `0.8.0a49`.

    `ClassificationMetric` accumulates a float64 confusion matrix, and MPS has no float64 -
    so `accuracy`, `balanced_accuracy` and `f1` have raised on every Apple Silicon Mac since
    the task landed. Nothing caught it because the default metric is `auc`, which takes the
    scores path and was already moving its state to the CPU for a different reason
    (memory), and no macOS test asked for one of the other three. It was found by the
    multi-label family hitting the identical line in `pyplatypus#315`.

    Asserted as an ordering rather than as a dtype because no local device can refuse the
    conversion: this machine has CUDA, CUDA has float64, and that is the whole reason the
    suite was green here while macOS was not. The sibling test in
    `test_multilabel_classification_metrics.py` has the longer version.
    """

    class Recorded(torch.overrides.TorchFunctionMode):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    metric = build_classification_metric(AccuracyMetric())
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 3.0], [3.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 0])

    recorder = Recorded()
    with recorder:
        metric.update(logits, labels)

    assert "cpu" in recorder.calls, "the matrix is being built where the batch lives"
    assert recorder.calls.index("cpu") < recorder.calls.index("double")
    assert metric._confusion.device.type == "cpu"
    assert metric._confusion.dtype == torch.float64
    assert metric.compute() == pytest.approx(1.0)
