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

**There are two families here and `multilabel` chooses between them rather than switching
one.** A label that is a *set* - `labels_from`, where every class is an independent
question - has no confusion matrix at all: an image with three findings is not a row of
one. So the state is a two-by-two table per class, the hard prediction is per class rather
than an `argmax`, and `accuracy` is **absent** rather than redefined, which is measured:
a model answering no to every class scores above one holding half the available signal on
both readings of accuracy, while `balanced_accuracy`, `f1` and `auc` all put it below every
model with any signal. `MultiLabelMetric` carries the table and `#307` the working.

What the two share is the protocol and not the state, which is why `_OverAPass` holds the
first and neither holds the second. `_auc` is shared outright: one class against the rest
is the same statistic whether the rest are alternatives or other questions.
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


class _OverAPass(nn.Module):
    """Reset, update, compute - the protocol, with no state of its own.

    Here because there are now two families that follow it and they do not share a *state*:
    one label per image accumulates a confusion matrix, and a label that is a set
    accumulates a two-by-two table per class. What they do share is that the number only
    exists once the pass is over, and that asking for it in one call has to give the same
    answer as adding batch by batch - which is the whole of what this class holds.
    """

    name = "metric"

    #: Whether `update` has to keep every example's probability rather than only counts.
    #: True for `auc` alone in both families. See this module's own first paragraphs for
    #: what it costs and for why it is not also a field on the specification.
    needs_scores = False

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

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


class ClassificationMetric(_OverAPass):
    """Higher is better, always in 0..1.

    `forward` is the one-shot form - reset, update, compute - which is what scoring a whole
    split in one call looks like. A training pass calls the three separately, because an
    epoch arrives in batches and the number only exists once all of them have.
    """

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
        #
        # **Counted on the CPU, and that is a different reason from the scores above.**
        # There the transfer is about memory; here it is that the accumulator is `float64`
        # and **MPS has no float64 at all** - `Cannot convert a MPS Tensor to float64
        # dtype`. Every Apple Silicon Mac picks `mps` since `0.8.0a38` repaired
        # `pick_device`, so this path raises there for any metric that reads a confusion
        # matrix rather than scores: `accuracy`, `balanced_accuracy` and `f1`. It costs one
        # int tensor of the batch's length per batch, which is nothing, and it also means
        # the bincount itself never has to exist on the accelerator.
        flat = (labels * n_class + predicted).cpu()
        counts = torch.bincount(flat, minlength=n_class * n_class).reshape(n_class, n_class)
        if self._confusion is None:
            self._confusion = counts.double()
        else:
            self._confusion += counts.double()

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


def _checked_multilabel(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The target as a boolean `(examples, classes)`, or a refusal naming the shapes.

    A multi-hot target has the shape of the logits, which is the one way it differs from a
    single label's: there is no index form to accept, because a set of labels has no single
    index. So this refuses rather than guessing - a `(examples,)` target here is a
    single-label target handed to the wrong family, and reading it as one row of a multi-hot
    would be a plausible answer to a question nobody asked.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"a label's logits are (examples, classes) and these are {tuple(logits.shape)}; "
            "a classifier pools its features before the head, so there is no spatial axis "
            "left by the time a metric sees them"
        )
    if tuple(target.shape) != tuple(logits.shape):
        raise ValueError(
            f"a multi-label target is one column per class, so {tuple(logits.shape)} of "
            f"logits needs a target of the same shape and this is {tuple(target.shape)}. A "
            "target of class indices is the single-label shape: these metrics are built "
            "from `labels_from`, where every class is an independent question."
        )
    return target > 0


class MultiLabelMetric(_OverAPass):
    """Higher is better, always in 0..1 - over a label that is a set.

    **One two-by-two table per class rather than one confusion matrix, and that is the whole
    reason this is a second family instead of a flag.** A confusion matrix answers *which
    one of these was it*, and with independent questions there is no such quantity: an image
    with three findings is not a row of that matrix. So the state here is `(4, classes)` -
    true positives, false positives, false negatives and true negatives per class - and
    every metric below reads it.

    **The hard prediction is `logits > 0`, which is `sigmoid(logits) > 0.5` exactly without
    computing the sigmoid**, and the point is that each class decides for itself. An
    `argmax` here would hand back one class per image, which is the answer the model was
    not asked for.

    There is no `accuracy` in this family, and that is measured rather than stylistic.
    Neither reading of it survives: on 4000 images at a prevalence of 10%, 5%, 3% and 2%,
    with one draw of the truth and models whose ordering is degraded by noise rather than
    shrunk,

        quality        exact   per-class   balanced   macro f1   macro auc
        1.00          0.8353      0.9570     0.5742     0.2339      0.9029
        0.50          0.8023      0.9477     0.5433     0.1444      0.8041
        0.00          0.7650      0.9370     0.4984     0.0184      0.4834
        all-negative  0.8137      0.9507     0.5000     0.0000      0.5000

    **a model that answers no to everything scores above the model holding half the
    available signal** - 0.8137 against 0.8023 on exact match, and 0.9507 against 0.9477
    per class - while the three metrics here all put it below every model with any signal
    at all. A number that rewards giving up cannot be what a `monitor` watches. Note that
    this is a *different* failure from the single-label one: there, accuracy is nearly flat
    and non-monotone in the signal; here both readings are monotone, and what breaks them is
    the comparison against a collapse, which is not a point on that curve.
    """

    def reset(self) -> None:
        self._tables: torch.Tensor | None = None
        self._scores: list[torch.Tensor] = []
        self._present: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        present = _checked_multilabel(logits, target)
        if self.needs_scores:
            # Off the accelerator, for the reason the single-label family gives: this is the
            # one piece of state whose size grows with the pass.
            self._scores.append(logits.sigmoid().detach().cpu())
            self._present.append(present.detach().cpu())
            return
        # On the CPU for the single-label family's second reason rather than its first: the
        # table is `float64` and MPS has none, so building it where the batch lives raises
        # on every Apple Silicon Mac. Two batches of booleans is what crosses, which is
        # smaller than the logits that produced them.
        chosen = (logits > 0).cpu()
        present = present.cpu()
        counts = torch.stack(
            [
                (chosen & present).sum(dim=0),
                (chosen & ~present).sum(dim=0),
                (~chosen & present).sum(dim=0),
                (~chosen & ~present).sum(dim=0),
            ]
        ).double()
        self._tables = counts if self._tables is None else self._tables + counts

    def _counts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """True positives, false positives, false negatives and true negatives per class.

        Refused rather than defaulted when nothing was added, which is the single-label
        family's reasoning: the comfortable answer is a zero that reads as a model getting
        everything wrong.
        """
        if self._tables is None:
            raise ValueError(
                f"`{self.name}` was asked for a value before any batch was added to it"
            )
        return tuple(self._tables)  # type: ignore[return-value]

    def _needs_both(self, positives: torch.Tensor, negatives: torch.Tensor) -> None:
        """Refuse a class the pass holds only one side of.

        A per-class recall needs an example of the class and a specificity needs one of
        something else, so a finding that is absent from every image - or present in every
        one - has no figure to contribute. Skipping it is the alternative and it is worse
        for the reason it is worse in the other family: a mean over the classes that
        happened to be answerable is a different quantity under the same name.
        """
        for label, counted in (("no positive", positives), ("no negative", negatives)):
            empty = (counted == 0).nonzero().flatten().tolist()
            if empty:
                raise ValueError(
                    f"`{self.name}` averages over classes and {empty} have {label} in this "
                    "pass, so there is nothing to average for them. With independent "
                    "questions this is a class nobody labelled either way rather than a "
                    "split that lost one, and `class_balance` is where to look."
                )


class MultiLabelBalancedAccuracy(MultiLabelMetric):
    """Sensitivity and specificity per class, averaged within a class and then across.

    The within-class average is what makes a rare finding weigh as much as a common one at
    all: on 4000 images at these prevalences a class's own accuracy is 0.95 whatever the
    model did, and this reads 0.5000 for the all-negative answer by construction - which is
    the control the measurement in `MultiLabelMetric` turns on.
    """

    name = "balanced_accuracy"

    def compute(self) -> float:
        true_positive, false_positive, false_negative, true_negative = self._counts()
        self._needs_both(true_positive + false_negative, true_negative + false_positive)
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        return ((sensitivity + specificity) / 2).mean().item()


class MultiLabelMacroF1(MultiLabelMetric):
    """Per-class F1, averaged over classes - the same formula one task across.

    Called through `dice_from_overlaps` for the reason the single-label one is: F1 is what
    Dice is called when the pixel is an image, and a second copy of `2*tp/(2*tp+fp+fn)`
    would be a second place for one formula to live. `smooth` is 0, so a class the pass
    never held is refused rather than flattered to 1.0.

    It needs a positive of each class and not a negative, which is the one place this family
    asks for less than `balanced_accuracy` does: a finding present in every image has a
    perfectly good F1 and no specificity.
    """

    name = "f1"

    def compute(self) -> float:
        true_positive, false_positive, false_negative, _ = self._counts()
        positives = true_positive + false_negative
        empty = (positives == 0).nonzero().flatten().tolist()
        if empty:
            raise ValueError(
                f"`{self.name}` averages over classes and {empty} appear in no example of "
                "this pass, so there is nothing to average for them."
            )
        return (
            f.dice_from_overlaps(true_positive, false_positive, false_negative, smooth=0.0)
            .mean()
            .item()
        )


class MultiLabelAuc(MultiLabelMetric):
    """The area under the ROC curve per class, averaged - and the default here too.

    **The ranking is the same whether it is read off the probability or off the logit**,
    which is the one thing that is simpler here than with a softmax: a sigmoid is monotone
    in its own logit and in nothing else, so the two orderings are identical and the AUCs
    agree to 0.00e+00. The single-label family's are 0.0866 apart over two classes, because
    a softmax is not monotone in one logit on its own. The probability is still what is
    stored, because it is what a caller is handed and what a calibration curve is drawn
    from.

    `_auc` is shared with the single-label family unchanged: one class against the rest is
    the same statistic whether the rest are alternatives or other questions.
    """

    name = "auc"
    needs_scores = True

    def compute(self) -> float:
        if not self._scores:
            raise ValueError(f"`{self.name}` was asked for a value before any batch was added")
        scores = torch.cat(self._scores).double()
        present = torch.cat(self._present)
        per_class = [
            _auc(scores[:, klass], present[:, klass].long()) for klass in range(scores.shape[1])
        ]
        return sum(per_class) / len(per_class)


_BUILDERS = {
    AccuracyMetric: Accuracy,
    BalancedAccuracyMetric: BalancedAccuracy,
    F1Metric: MacroF1,
    AucMetric: Auc,
}


#: The same four names, where three of them mean something when a label is a set. Read by
#: `build_classification_metric` and held to the specification's own `supports_multilabel`
#: by `tests/objectives/test_multilabel_classification_metrics.py`, which is the pair: the
#: flag is what the configuration layer refuses on, and this mapping is what exists.
_MULTILABEL_BUILDERS = {
    BalancedAccuracyMetric: MultiLabelBalancedAccuracy,
    F1Metric: MultiLabelMacroF1,
    AucMetric: MultiLabelAuc,
}


def build_classification_metric(
    spec: ClassificationMetricSpec, *, multilabel: bool = False
) -> _OverAPass:
    """`multilabel` and no `spacing`, which is one of the two `build_metric` takes.

    There is no grid, so there is no tolerance in millimetres to convert. There *are*
    independent channels when the labelling named one column per class, and then every one
    of these is a different quantity computed from different state - so this returns a
    different object rather than setting a flag on one, which is where it differs from
    `build_metric` as well.

    `multilabel` is a property of the data and is passed in by whoever built the loader,
    never read off a target: a column of all zeros is ordinary in both worlds, so there is
    nothing in the tensors to tell them apart.
    """
    builders = _MULTILABEL_BUILDERS if multilabel else _BUILDERS
    builder = builders.get(type(spec))
    if builder is None:
        if multilabel and type(spec) in _BUILDERS:
            # Reached only by a caller that built a metric list without the specification,
            # because `ClassificationSpec` refuses this by name first - the flag on the
            # specification is the one place the answer lives and this is the second line
            # of a wall rather than a second copy of it.
            raise KeyError(
                f"`{spec.name}` is not admissible when a label is a set: a model answering "
                "no to every class scores above one with half the signal on both readings "
                "of it, so it would reward a collapse. `balanced_accuracy`, `f1` and `auc` "
                "all put that answer below every model that has any signal."
            )
        raise KeyError(f"no implementation for metric {type(spec).__name__}")  # pragma: no cover
    return builder()


def build_classification_metrics(
    specs: list[ClassificationMetricSpec], *, multilabel: bool = False
) -> dict[str, _OverAPass]:
    """Keyed by name, because that is how they end up in the history table.

    The key is the same for either family - `auc` is `auc` in a history column whether the
    label was one of a set or a set - which is what lets a callback watch `val_auc` without
    knowing how the classes were named.
    """
    return {spec.name: build_classification_metric(spec, multilabel=multilabel) for spec in specs}
