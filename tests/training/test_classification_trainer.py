"""Training a classifier, and the three things an epoch must not get wrong.

The loss is a mean over examples and has to be exact however the pass was batched. The
metrics are not means over examples at all, so they are accumulated and reset per pass. And
what the epoch puts in its logs has to be what the specification said a callback may watch -
`0.8.0a27` is what the other answer costs, one task across.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pyplatypus.objectives import functional as f
from pyplatypus.objectives.classification import ClassificationMetric, MultiLabelMetric
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.spec.components import AccuracyMetric, AucMetric
from pyplatypus.training.callbacks import Callback
from pyplatypus.training.classification_trainer import ClassificationTrainer


class Fixed(nn.Module):
    """The same logits for every image, so a reported number can be checked by hand.

    One real parameter, so there is something for the optimiser to step and the run is a
    run - but the logits do not depend on it, which is what keeps the arithmetic stable
    across epochs.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits[: x.shape[0]] + 0.0 * self.unused


class FromImages(nn.Module):
    """Logits read out of the image, so the same images give the same answer however they
    are cut into batches - which is what makes a batching comparison about the pooling
    rather than about the fixture."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x.mean(dim=(1, 2, 3))
        return torch.stack([value, -value], dim=1) + 0.0 * self.unused


class Learns(nn.Module):
    """Logits driven by one parameter, so training moves the answer."""

    def __init__(self, n_class: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.n_class = n_class

    def forward(self, x):
        base = torch.arange(self.n_class, dtype=torch.float32).repeat(x.shape[0], 1)
        return base * (1.0 + self.weight)


def spec(**over) -> ClassificationModel:
    return ClassificationModel(
        **{"name": "c", "input_shape": (8, 8), "blocks": 2, "filters": 2, "epochs": 1, **over}
    )


def images(n: int) -> torch.Tensor:
    return torch.zeros(n, 3, 8, 8)


# --------------------------------------------------------------------------- the loss


def test_the_loss_is_the_one_hot_cross_entropy_and_not_the_index_one():
    """The trap this expansion exists for, and the control that it is the right expansion.

    This package's `cross_entropy` is written for a one-hot target. Handing it an index
    raises when the batch size differs from the class count, and with a batch of 2 against
    2 classes it returns a number that is not the loss. So the number the epoch reports is
    compared against torch's own index form, which is the independent implementation.
    """
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 0, 1])
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])

    reported = trainer.evaluate([(images(4), labels)])["val_loss"]
    expected = torch.nn.functional.cross_entropy(logits, labels).item()
    assert reported == pytest.approx(expected, abs=1e-12)


def test_the_shape_that_would_have_hidden_it():
    """A batch of two against two classes is where an index target stops raising and starts
    answering. Measured before the fix: 1.126928 where the loss is 0.126928."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])
    reported = trainer.evaluate([(images(2), labels)])["val_loss"]

    assert reported == pytest.approx(0.126928, abs=1e-6)
    assert reported != pytest.approx(1.126928, abs=1e-6)


def test_the_epoch_loss_does_not_depend_on_how_the_pass_was_batched():
    """Weighted by each batch's size, so a ragged last batch is not worth a whole one -
    which is `maju116/rplatypus#196`'s repair, in the task where a case is one image.

    `metrics=[]` because this test is about the loss: three of the four metrics are
    deliberately **not** means over examples, so including one would be asking this
    question of something that answers a different one.
    """
    torch.manual_seed(3)
    pictures = torch.rand(5, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 0])

    def loss_over(batches):
        trainer = ClassificationTrainer(
            FromImages(), spec(metrics=[]), n_class=2, device="cpu", callbacks=[]
        )
        return trainer.evaluate(batches)["val_loss"]

    whole = loss_over([(pictures, labels)])
    ragged = loss_over(
        [(pictures[:2], labels[:2]), (pictures[2:4], labels[2:4]), (pictures[4:], labels[4:])]
    )
    even = loss_over([(pictures[:1], labels[:1]), (pictures[1:], labels[1:])])

    assert ragged == pytest.approx(whole, abs=1e-6)
    assert even == pytest.approx(whole, abs=1e-6)
    # and the control: a mean over BATCHES rather than examples would not agree
    by_batch = (
        loss_over([(pictures[:2], labels[:2])])
        + loss_over([(pictures[2:4], labels[2:4])])
        + loss_over([(pictures[4:], labels[4:])])
    ) / 3
    assert by_batch != pytest.approx(whole, abs=1e-4)


# ------------------------------------------------------------------------ the metrics


def test_the_metrics_are_reset_at_the_start_of_every_pass():
    """Carried over, epoch 10 would be a weighted average of ten epochs and the figure would
    improve for no reason the model had anything to do with."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    trainer = ClassificationTrainer(
        Fixed(logits),
        spec(epochs=3, metrics=[{"name": "accuracy"}]),
        n_class=2,
        device="cpu",
        callbacks=[],
    )
    history = trainer.fit([(images(2), labels)], [(images(2), labels)], epochs=3)
    accuracies = [row["val_accuracy"] for row in history.records]
    assert accuracies == [1.0, 1.0, 1.0], accuracies
    # and the accumulator is empty between passes rather than holding the last one
    assert trainer.metrics["accuracy"]._confusion.sum().item() == 2


def test_an_epoch_reports_what_the_whole_pass_says_rather_than_a_batch_mean():
    """The protocol slice 4 measured, end to end through the trainer: the figure is the one
    the pass produces, which for three of the four metrics is not the mean of the batches'."""
    from pyplatypus.objectives.classification import Auc

    torch.manual_seed(0)
    logits = torch.randn(40, 2)
    labels = (torch.rand(40) < 0.25).long()
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])

    batched = trainer.evaluate([(images(8), labels[i : i + 8]) for i in range(0, 40, 8)])
    # Fixed() returns `logits[:n]` per batch, so build the direct answer the same way
    stacked = torch.cat([logits[: len(labels[i : i + 8])] for i in range(0, 40, 8)])
    relabelled = torch.cat([labels[i : i + 8] for i in range(0, 40, 8)])
    assert batched["val_auc"] == pytest.approx(Auc()(stacked, relabelled).item(), abs=1e-12)


def test_a_pass_holding_none_of_a_class_is_refused_rather_than_scored():
    """The honest answer, and the reason the engine owes a stratified split: measured, a
    random 70/15/15 split of 100 images at 3% prevalence leaves some part empty of the rare
    class 90% of the time."""
    logits = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    trainer = ClassificationTrainer(
        Fixed(logits),
        spec(metrics=[{"name": "balanced_accuracy"}]),
        n_class=2,
        device="cpu",
        callbacks=[],
    )
    with pytest.raises(ValueError, match=r"\[1\]"):
        trainer.evaluate([(images(2), torch.tensor([0, 0]))])


# --------------------------------------------------- what the spec promised a callback


def test_the_epoch_produces_exactly_what_the_specification_says_is_watchable():
    """The pair, and it is the one `0.8.0a27` was written for.

    `ClassificationModel.monitorable` is what refuses a callback when the configuration is
    read; the trainer's epoch is what actually puts keys in `state.logs`. Two layers
    answering one question, so a test asserts the answers agree - over every metric the
    task offers rather than over the default, because a pairing proves nothing on a set of
    one.
    """
    from typing import get_args

    from pyplatypus.spec.components import ClassificationMetricSpec

    members = get_args(get_args(ClassificationMetricSpec)[0])
    names = [member.model_fields["name"].default for member in members]
    assert len(names) > 1, "the selection is the instrument"

    model_spec = spec(metrics=[{"name": name} for name in names])
    torch.manual_seed(1)
    trainer = ClassificationTrainer(
        Fixed(torch.randn(6, 2)), model_spec, n_class=2, device="cpu", callbacks=[]
    )
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    history = trainer.fit([(images(6), labels)], [(images(6), labels)], epochs=1)

    produced = set(history.records[-1])
    bookkeeping = {"epoch", "seconds", "learning_rate"}
    assert model_spec.monitorable == produced - bookkeeping
    assert not any(key.endswith("_worst") for key in produced)
    assert not any("foreground" in key for key in produced)


def test_a_callback_watching_a_metric_stops_the_run():
    """The epoch's keys reaching a callback, rather than the callback being asked to trust
    the specification."""

    class StopOnSecond(Callback):
        def __init__(self):
            self.seen = []

        def on_epoch_end(self, state) -> bool:
            self.seen.append(state.logs["val_accuracy"])
            state.stop_reason = "told to"
            return len(self.seen) == 2

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    watcher = StopOnSecond()
    trainer = ClassificationTrainer(
        Fixed(logits),
        spec(epochs=5, metrics=[{"name": "accuracy"}]),
        n_class=2,
        device="cpu",
        callbacks=[watcher],
    )
    history = trainer.fit([(images(2), labels)], [(images(2), labels)], epochs=5)
    assert len(history.records) == 2
    assert history.stop_reason == "told to"
    assert watcher.seen == [1.0, 1.0]


def test_a_run_trains_something():
    """One real parameter and a loss that can fall, because a trainer that reports numbers
    without moving any weights would pass everything above."""
    model = Learns()
    before = model.weight.detach().clone()
    trainer = ClassificationTrainer(
        model,
        spec(epochs=3, metrics=[], optimizer={"name": "adam", "learning_rate": 0.5}),
        n_class=2,
        device="cpu",
        callbacks=[],
    )
    labels = torch.tensor([0, 0, 0, 0])
    history = trainer.fit([(images(4), labels)] * 3, epochs=3)
    assert not torch.equal(model.weight.detach(), before)
    assert history.records[-1]["train_loss"] < history.records[0]["train_loss"]


# ------------------------------------------------------------------- the transfer


class WithTransfer(nn.Module):
    """A model shaped like a pretrained one, without a network.

    The trainer reads `model.encoder.transferred`, which is what `PretrainedEncoder` sets
    for the layers that arrived with somebody else's weights - so a stub with that attribute
    exercises the freeze offline, and whether the weights are ImageNet's or random is
    irrelevant to the question of whether they are held still.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.transferred = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
        self.ours = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x.mean(dim=(1, 2, 3))
        return torch.stack([value, -value], dim=1) + 0.0 * self.ours


def transfer_spec(**over) -> ClassificationModel:
    """No `blocks` or `filters`: the spec refuses our knobs beside somebody else's backbone,
    which is slice 2's validator and is why this helper exists rather than `spec(...)`."""
    return ClassificationModel(
        **{"name": "t", "input_shape": (32, 32), "encoder": "resnet34", "metrics": [], **over}
    )


def test_the_transferred_layers_are_held_still_for_as_long_as_asked():
    """Both halves, and the second is the one the docstring says is easy to miss: a frozen
    BatchNorm still rewrites its running statistics from every batch it sees unless the
    module is also in `eval()`, and pretrained weights were fitted beside pretrained
    statistics.
    """
    model = WithTransfer()
    seen: list[tuple[int, bool, bool]] = []

    class Watch(Callback):
        def on_epoch_end(self, state) -> bool:
            transferred = model.encoder.transferred
            seen.append(
                (
                    state.epoch,
                    all(not p.requires_grad for p in transferred.parameters()),
                    transferred.training,
                )
            )
            return False

    trainer = ClassificationTrainer(
        model,
        transfer_spec(epochs=3, freeze_encoder=2),
        n_class=2,
        device="cpu",
        callbacks=[Watch()],
    )
    assert trainer.transferred is model.encoder.transferred, "the trainer found the transfer"
    trainer.fit([(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))], epochs=3)

    assert seen == [(1, True, False), (2, True, False), (3, False, True)]


def test_a_model_with_nothing_transferred_is_left_alone():
    """The control: without an `encoder` there is no transfer, so the freeze is a no-op
    rather than something reaching for an attribute that is not there."""
    trainer = ClassificationTrainer(
        Learns(), spec(metrics=[]), n_class=2, device="cpu", callbacks=[]
    )
    assert trainer.transferred is None
    trainer._set_transferred_frozen(True)
    assert trainer._frozen is None


# ------------------------------------------------------------------------- answering


def test_predict_returns_probabilities_over_the_classes():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])
    out = trainer.predict([(images(3), torch.tensor([0, 1, 0]))])

    assert out.shape == (3, 2)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out, logits.softmax(dim=1).numpy(), atol=1e-6)


def test_predict_streams_a_batch_at_a_time_and_accepts_an_unlabelled_loader():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])
    pieces = list(trainer.predict_stream([images(2), images(2)]))
    assert [piece.shape for piece in pieces] == [(2, 2), (2, 2)]


def test_predict_labels_reads_the_names_in_class_order():
    """And refuses a list of the wrong length, which is the one mistake here that would
    otherwise produce plausible output."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    trainer = ClassificationTrainer(Fixed(logits), spec(), n_class=2, device="cpu", callbacks=[])
    loader = [(images(2), torch.tensor([0, 1]))]

    assert trainer.predict_labels(loader, ["benign", "malignant"]) == ["benign", "malignant"]
    assert trainer.predict_labels(loader, ["malignant", "benign"]) == ["malignant", "benign"]
    with pytest.raises(ValueError, match="3 names were given"):
        trainer.predict_labels(loader, ["a", "b", "c"])


def test_an_empty_loader_is_refused_rather_than_divided_by():
    trainer = ClassificationTrainer(
        Fixed(torch.zeros(1, 2)), spec(), n_class=2, device="cpu", callbacks=[]
    )
    with pytest.raises(ValueError, match="produced no batches"):
        trainer.evaluate([])


# ------------------------------------------------------------- a label that is a set


def multi_spec(**over) -> ClassificationModel:
    """Four classes and `auc`, which is the default and the one metric that crosses with
    its definition untouched - macro-OvR equals the binary AUC to every digit."""
    return ClassificationModel(
        **{
            "name": "c",
            "input_shape": (8, 8),
            "blocks": 2,
            "filters": 2,
            "epochs": 1,
            "metrics": [AucMetric()],
            **over,
        }
    )


def multi_trainer(logits: torch.Tensor, **over) -> ClassificationTrainer:
    return ClassificationTrainer(
        Fixed(logits),
        multi_spec(**over),
        n_class=logits.shape[1],
        multilabel=True,
        device="cpu",
        callbacks=[],
    )


FINDINGS = ["atelectasis", "cardiomegaly", "effusion", "nodule"]


def test_the_loss_is_the_binary_form_and_the_target_crosses_untouched():
    """A set of labels has no index form, so there is nothing to expand - and the control
    is the same logits and the same target through the categorical route, which gives a
    different number for the same arrays."""
    logits = torch.tensor([[2.0, -1.0, 0.5, -3.0], [-2.0, 1.0, -0.5, 3.0]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    trainer = multi_trainer(logits)

    assert trainer.loss_fn.multilabel is True
    reported = trainer.evaluate([(images(2), target)])["val_loss"]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target).item()
    assert reported == pytest.approx(expected, abs=1e-12)

    # No metrics on the control, because the single-label `auc` reads a multi-hot target as
    # a one-hot one and refuses a class with no positive - which is the right refusal and
    # not the thing being compared here.
    categorical = ClassificationTrainer(
        Fixed(logits), multi_spec(metrics=[]), n_class=4, device="cpu", callbacks=[]
    ).evaluate([(images(2), target)])["val_loss"]
    assert categorical != pytest.approx(reported, abs=1e-6), (
        "the control: the two routes are different objectives on the same arrays"
    )


def test_a_target_of_the_wrong_width_is_refused_rather_than_broadcast():
    """Measured rather than assumed, and the assumption was wrong twice.

    The first guess was that a wrong-width target would *broadcast*: it does not, torch
    refuses it by name, including at the square shape that defeats the categorical form. The
    second was that `_loss_of`'s multi-label branch therefore changes no number - deleting
    it makes this test pass a `(3,)` target to `as_onehot`, which reads it as **indices**,
    expands it into a plausible one-hot, and the binary loss returns 0.843 instead of
    refusing. So this is what makes that branch load-bearing rather than declarative.
    """
    square = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    indices = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match="must be the same as input size"):
        multi_trainer(square).evaluate([(images(3), indices.float())])

    # What the branch is standing between: the expansion that rescues the categorical route
    # turns this target into something the binary loss accepts without complaint.
    rescued = f.as_onehot(indices.float(), 3)
    assert tuple(rescued.shape) == (3, 3)
    assert f.binary_cross_entropy(square, rescued).item() == pytest.approx(0.842927, abs=1e-6)

    # The control, one task over: the same shape through the categorical route produces a
    # number. That is the trap `as_onehot` is in `_loss_of` for.
    through_cce = ClassificationTrainer(
        Fixed(square), multi_spec(), n_class=3, device="cpu", callbacks=[]
    ).evaluate([(images(3), indices)])["val_loss"]
    assert through_cce > 0


def test_the_metrics_are_the_two_by_two_family_rather_than_a_confusion_matrix():
    logits = torch.tensor([[2.0, -1.0, 0.5, -3.0], [-2.0, 1.0, -0.5, 3.0]])
    assert isinstance(multi_trainer(logits).metrics["auc"], MultiLabelMetric)
    assert isinstance(
        ClassificationTrainer(
            Fixed(logits), multi_spec(), n_class=4, device="cpu", callbacks=[]
        ).metrics["auc"],
        ClassificationMetric,
    )


def test_accuracy_is_refused_when_the_label_is_a_set():
    """By name, where the metric list is built - the measurement is that a model answering
    no to everything scores above one holding half the signal, on both readings of it."""
    logits = torch.zeros(2, 4)
    with pytest.raises(KeyError, match="reward a collapse"):
        multi_trainer(logits, metrics=[AccuracyMetric()])


def test_the_probabilities_are_a_sigmoid_per_class_rather_than_a_softmax():
    logits = torch.tensor([[2.0, -1.0, 0.5, -3.0], [-2.0, 1.0, -0.5, 3.0]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    out = multi_trainer(logits).predict([(images(2), target)])

    assert np.allclose(out, logits.sigmoid().numpy(), atol=1e-6)
    assert not np.allclose(out.sum(axis=1), 1.0), (
        "independent questions do not sum to one; a softmax here would make them"
    )


def test_predict_labels_is_a_threshold_and_answers_with_a_list_per_image():
    """`0.5` of a probability is `logits > 0` exactly, which is the same decision the metric
    family makes on the hard prediction - so a score and an answer cannot disagree.

    Two rows are worth having beyond the ordinary one. The third is every logit negative, so
    the answer is the empty list rather than the least-negative class, which is what an
    `argmax` would have said. The fourth carries a logit of **exactly 0**, which is a
    probability of exactly 0.5 - the boundary where `> 0.5` and `>= 0.5` stop agreeing, and
    the metric's own rule is the strict one.
    """
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, -3.0],
            [-2.0, 1.0, -0.5, 3.0],
            [-4.0, -1.0, -2.0, -3.0],
            [0.0, 1.0, -1.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    answered = multi_trainer(logits).predict_labels([(images(4), target)], FINDINGS)

    assert answered == [
        ["atelectasis", "effusion"],
        ["cardiomegaly", "nodule"],
        [],
        ["cardiomegaly"],
    ]
    # The agreement with the hard prediction the metrics read, asserted rather than argued.
    for row, names in zip(logits, answered):
        assert [FINDINGS[i] for i in range(4) if row[i] > 0] == names

    with pytest.raises(ValueError, match="3 names were given"):
        multi_trainer(logits).predict_labels([(images(4), target)], ["a", "b", "c"])


def test_an_epoch_over_a_set_of_labels_reports_the_loss_and_every_metric_asked_for():
    """The join: a target that is a set through the loss, the metric family and the logs a
    callback is allowed to watch. `0.8.0a27` is what the last of those costs when it is
    wrong, one task across."""
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, -3.0],
            [-2.0, 1.0, -0.5, 3.0],
            [1.0, -2.0, -1.0, 2.0],
            [-1.0, 2.0, 1.0, -2.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    trainer = multi_trainer(logits)
    history = trainer.fit([(images(4), target)], [(images(4), target)])

    logs = history.records[-1]
    assert {"train_loss", "val_loss", "train_auc", "val_auc"} <= set(logs)
    watchable = trainer.spec.monitorable
    assert {key for key in logs if key.startswith(("train_", "val_"))} <= watchable
    assert 0.0 <= logs["val_auc"] <= 1.0
