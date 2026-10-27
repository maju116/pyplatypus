"""Metrics over a label, where a batch is not a unit a score can be read off.

Everything in `objectives.metrics` is a formula on overlap counts, and the trainer reports
one by averaging what each batch returned, weighted by how many examples the batch held.
That is exact for a mean over examples and wrong for everything else - which is most of
what anybody wants to know about a classifier. Measured on 4000 images at a prevalence of
3%, against a model whose own probabilities are the truth:

    pooled over the pass    batch 8    batch 32   batch 128
    accuracy      0.9742     0.9742     0.9742     0.9742     exact, to every digit
    balanced      0.5700    +0.0035    +0.0047    +0.0123
    macro f1      0.6120    -0.0734    -0.0435    -0.0041
    auc                      off by 0.0008 to 0.0371, and not monotone in the batch size

The reason is one reason for all three: at that prevalence **398 of 500 batches of 8 hold
no positive at all**, so a per-class recall, an F1 and a rank statistic are each a mean over
whichever batches happened to contain the minority class. A bigger batch does not fix it -
the AUC's error is larger at 128 than at 32 - because the quantity is not a mean over
batches in the first place.

So these are accumulated and computed once: `update` per batch, `compute` when the pass is
over. `accumulates` asks whether a tiled image's score can be rebuilt from its tiles'
counts, and that is not this question - there are no tiles and the unit is the pass.

**What has to be retained is the one thing that differs between them.** Three of the four
need a confusion matrix, which is `n_class` squared numbers however long the pass is; `auc`
needs every example's probability, because a ranking cannot be summarised. For 4000 images
and two classes that is 4 numbers against 8000, and `needs_scores` is where a metric says
which it is. It decides how much is held and nothing else - there is no configuration it
refuses - so the specification layer deliberately carries no copy of it. A flag mirrored
into a layer that asks nothing of it is the declaration beside a mechanism that §3F of the
workbench's notes is a list of.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.objectives import functional as f
from pyplatypus.spec.components import (
    AccuracyMetric,
    AucMetric,
    BalancedAccuracyMetric,
    ClassificationMetricSpec,
    F1Metric,
)


def _probabilities_and_labels(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(n, n_class)` probabilities and `(n,)` class indices, from either shape of target.

    The target arrives as indices from a reader that read a directory name and as one-hot
    from anything that built it the way a mask is built, so both are accepted here rather
    than at each call site. `as_onehot` is the other direction and is not reused: it
    answers in one-hot and every metric below counts, which wants an index.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"a label's logits are (examples, classes) and these are {tuple(logits.shape)}; "
            "a classifier pools its features before the head, so there is no spatial axis "
            "left by the time a metric sees them"
        )
    labels = target.argmax(dim=1) if target.ndim == 2 else target
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"{logits.shape[0]} examples of logits against {tuple(target.shape)} of target"
        )
    return logits.softmax(dim=1), labels.long()


class ClassificationMetric(nn.Module):
    """Higher is better, always in 0..1.

    `forward` is the one-shot form - reset, update, compute - which is what scoring a whole
    split in one call looks like. A training pass calls the three separately, because an
    epoch arrives in batches and the number only exists once all of them have.
    """

    name = "metric"

    #: Whether `update` has to keep every example's probability rather than only counts.
    #: True for `auc` alone. See this module's own first paragraphs for what it costs and
    #: for why it is not also a field on the specification.
    needs_scores = False

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self._confusion: torch.Tensor | None = None
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities, labels = _probabilities_and_labels(logits, target)
        n_class = probabilities.shape[1]
        if self.needs_scores:
            # Off the accelerator, because this is the one thing here whose size grows with
            # the pass: a million images of a hundred classes is 400 MB of float32, and it
            # would be 400 MB of the memory the next batch wants. The transfer is one small
            # tensor per batch.
            self._scores.append(probabilities.detach().cpu())
            self._labels.append(labels.detach().cpu())
            return
        predicted = probabilities.argmax(dim=1)
        # One bincount rather than a loop over classes: the pair (truth, predicted) is a
        # single index into a flattened matrix, which is what makes this O(batch) rather
        # than O(batch * classes).
        flat = labels * n_class + predicted
        counts = torch.bincount(flat, minlength=n_class * n_class).reshape(n_class, n_class)
        if self._confusion is None:
            self._confusion = counts.double()
        else:
            self._confusion += counts.double()

    def compute(self) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.reset()
        self.update(logits, target)
        # `float64` rather than the default dtype, which is what `torch.tensor(a_float)`
        # would give: these are counts and ranks rather than activations, and wrapping the
        # answer in a float32 rounded it enough to put 2.5e-08 between this one-shot form
        # and the same pass added batch by batch. Two routes to one number must not differ
        # because of the container the answer is handed back in.
        return torch.tensor(self.compute(), dtype=torch.float64)

    # ------------------------------------------------------------------ what compute reads

    def _matrix(self) -> torch.Tensor:
        """The confusion matrix, as `[truth, predicted]`.

        Refused rather than defaulted when nothing was added: a metric computed over an
        empty pass has no value, and the comfortable answer here would be a zero that reads
        as a model getting everything wrong.
        """
        if self._confusion is None:
            raise ValueError(
                f"`{self.name}` was asked for a value before any batch was added to it"
            )
        return self._confusion

    def _supported(self, matrix: torch.Tensor) -> torch.Tensor:
        """Per-class support, refusing a class the pass never contained.

        Skipping it is the alternative and it is the worse one: a mean over the classes
        that happened to appear is a different quantity under the same name, and the split
        missing one of its classes entirely is a thing the person running this wants told.
        """
        support = matrix.sum(dim=1)
        empty = (support == 0).nonzero().flatten().tolist()
        if empty:
            raise ValueError(
                f"`{self.name}` averages over classes and {empty} appear in no example of "
                "this pass, so there is nothing to average for them. A split that holds "
                "none of a class cannot report a per-class figure; `accuracy` is the one "
                "that does not need every class to be present."
            )
        return support


class Accuracy(ClassificationMetric):
    """The share of images whose label was right.

    **Not the default, and the reason is measured rather than received.** The usual warning
    is that accuracy is unbalanced; the sharper fact is that at a low prevalence it is
    blind. Measured on 40000 images at a prevalence of 3%, with the model's own
    probabilities as the truth so that only its discrimination varies:

        discrimination   accuracy   balanced   macro f1      auc
        none               0.9720     0.5000     0.4929   0.5132
        a = 0.5            0.9702     0.5000     0.4924   0.6393
        a = 1.0            0.9704     0.5004     0.4933   0.7516
        a = 2.0            0.9728     0.5609     0.5972   0.9088
        a = 4.0            0.9802     0.7443     0.7923   0.9763

    Accuracy spans 0.0100 where the AUC spans 0.4631, and it is **not monotone**: the model
    with no signal at all scores above two models that have one. It is offered because it is
    the number people ask for and the one a report is read against, not because it answers
    the question.
    """

    name = "accuracy"

    def compute(self) -> float:
        matrix = self._matrix()
        return (matrix.diagonal().sum() / matrix.sum()).item()


class BalancedAccuracy(ClassificationMetric):
    """Recall per class, averaged - so every class weighs the same whatever its share.

    It fixes what accuracy is usually accused of and **not** what the table above shows:
    at a prevalence of 3% a calibrated model never crosses the threshold, so balanced
    accuracy sees the same all-negative prediction a collapsed model makes and reports
    0.5000 for both. Measured there: a model with a real signal (AUC 0.6393) is +0.0000
    away from one with none. The property that separates them is being threshold-free, not
    being balanced, which is why `auc` is the default and this is not.
    """

    name = "balanced_accuracy"

    def compute(self) -> float:
        matrix = self._matrix()
        return (matrix.diagonal() / self._supported(matrix)).mean().item()


class MacroF1(ClassificationMetric):
    """Per-class F1, averaged over classes.

    **The formula is Dice's and is called through it rather than written again.** Measured:
    `dice_from_overlaps(tp, fp, fn, smooth=0)` and `2*tp/(2*tp+fp+fn)` agree exactly on
    every class of a three-class draw, which they must - a one-hot label vector is a mask
    of one pixel, and F1 is what Dice is called when the pixel is an image. So this is the
    same mechanism under the name its audience uses, rather than a second copy of it.

    `smooth` is 0 here where a segmentation metric defaults it to 1: a class absent from
    the pass is refused by `_supported` instead of being flattered to 1.0, and smoothing
    a count of images would move a reported figure for no reason a reader could see.

    Micro-averaging is deliberately not offered. For a single label per image it **is**
    accuracy - measured, 0.7760000228881836 both ways on a three-class draw - so a
    `average: micro` would put the same number in the table twice under two names.
    """

    name = "f1"

    def compute(self) -> float:
        matrix = self._matrix()
        support = self._supported(matrix)
        true_positive = matrix.diagonal()
        false_positive = matrix.sum(dim=0) - true_positive
        false_negative = support - true_positive
        return (
            f.dice_from_overlaps(true_positive, false_positive, false_negative, smooth=0.0)
            .mean()
            .item()
        )


class Auc(ClassificationMetric):
    """The area under the ROC curve, one class against the rest, averaged over classes.

    **It is the default for this task**, because it is the only one of the four measured
    above that orders models by their discrimination monotonically, and the only one that
    can tell a real signal from a collapse at a low prevalence. It needs no threshold,
    which is the property that matters: the others all read the hard prediction, and at a
    prevalence of 3% a calibrated model's hard prediction is all-negative whether it has
    learnt anything or not.

    **Two classes need no special case.** Macro one-vs-rest over two classes is the binary
    AUC exactly - measured, 0.766991500000 for both, and the two one-vs-rest values are
    the same number because one score is one minus the other.

    **It ranks by the probability rather than by the logit, and that is a choice with a
    size.** `softmax` is not monotone in one logit on its own, so the two give different
    numbers: measured on one draw at each width, 0.0866 apart over two classes, 0.0598 over
    three, 0.0367 over five and 0.0209 over ten. The probability is what the caller is
    handed and what a calibration curve is drawn from, so it is what the ordering is read
    off; the logit's ordering would ignore how sure the model was about everything else.
    """

    name = "auc"
    needs_scores = True

    def compute(self) -> float:
        if not self._scores:
            raise ValueError(f"`{self.name}` was asked for a value before any batch was added")
        scores = torch.cat(self._scores).double()
        labels = torch.cat(self._labels)
        per_class = [
            _auc(scores[:, klass], (labels == klass).long()) for klass in range(scores.shape[1])
        ]
        return sum(per_class) / len(per_class)


def _auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """One class against the rest, by ranks rather than by counting pairs.

    The rank form is the Mann-Whitney U statistic and costs a sort; counting pairs costs
    the product of the two class sizes, which is 400 million comparisons on a 40000-image
    pass. Ties take the mean of the ranks they span, which is what makes a model that
    reports the same probability twice score those two examples as a coin flip rather than
    in whatever order the sort happened to put them. The pairwise form is this function's
    test control rather than its implementation, and the two agree to 0.00e+00 on a draw
    built to be nothing but ties.
    """
    n_positive = int(positive.sum())
    n_negative = int(positive.numel() - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "an AUC needs both an example of the class and an example of something else, "
            f"and this pass has {n_positive} of one and {n_negative} of the other. A class "
            "absent from a split has no ranking to measure"
        )
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    positions = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    unique, inverse, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
    totals = torch.zeros(unique.numel(), dtype=torch.float64).scatter_add_(0, inverse, positions)
    ranks = (totals / counts)[inverse]
    of_the_class = ranks[positive[order] == 1].sum()
    return ((of_the_class - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)).item()


_BUILDERS = {
    AccuracyMetric: Accuracy,
    BalancedAccuracyMetric: BalancedAccuracy,
    F1Metric: MacroF1,
    AucMetric: Auc,
}


def build_classification_metric(spec: ClassificationMetricSpec) -> ClassificationMetric:
    """No `multilabel` and no `spacing`: neither means anything to a label.

    A label is one answer out of a set of alternatives, so there are no independent channels
    to decide for themselves, and there is no grid, so there is no tolerance in millimetres
    to convert. `build_metric` takes both because a mask has both.
    """
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover
        raise KeyError(f"no implementation for metric {type(spec).__name__}")
    return builder()


def build_classification_metrics(
    specs: list[ClassificationMetricSpec],
) -> dict[str, ClassificationMetric]:
    """Keyed by name, because that is how they end up in the history table."""
    return {spec.name: build_classification_metric(spec) for spec in specs}
