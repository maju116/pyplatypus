"""What every trainer owes, in as many copies as there are trainers.

Two contracts live here. One is the callback hooks - `maju116/pyplatypus#300` - and the
other is that a frozen transfer stays frozen, which is this file's second entry because
removing the mechanism that guarantees it broke **nothing** in the segmentation suite, in
released code, with the reasoning written out in a three-line comment beside it.

`on_batch_end` arrived in `0.8.0a37` as the fourth hook, with `ema` as the callback written
for it, and the second trainer was not given either it or `state.step`. `ema` is the only
callback that implements it, so a detection run configured with `ema` took no exponential
average at all - accepted by the specification, costing what it costs, and doing nothing.
Measured before the fix: 0 folds over four batches and two epochs, against 8 for the same
callback through the segmentation trainer.

**The hook contract is one mechanism in as many copies as there are trainers**, and the
epoch loops around it are genuinely not one mechanism - each task pools its numbers by
something different, which is why nobody compares them and why the drift was in `_run_epoch`
rather than in `fit`, where the two are character for character the same. So this is the
pair §3F of the workbench's notes describes: a test holding the copies together rather than
an attempt to make them one.

Both halves are read off the code rather than listed here. A third trainer is coming for
`task: classification`, and a trainer this file has no builder for **fails** rather than
being quietly left out - which is the half that makes the discovery an instrument instead of
a decoration.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from pyplatypus import training
from pyplatypus.detection.encode import encode
from pyplatypus.spec.classification import ClassificationModel
from pyplatypus.spec.detection import DetectionModel
from pyplatypus.spec.models import SegmentationModel
from pyplatypus.training.callbacks import Callback, TrainingState
from pyplatypus.training.classification_trainer import ClassificationTrainer
from pyplatypus.training.detection_trainer import DetectionTrainer
from pyplatypus.training.trainer import Trainer

ANCHORS = (((0.25, 0.25), (0.15, 0.30)), ((0.10, 0.10), (0.06, 0.12)), ((0.04, 0.04), (0.02, 0.05)))
DETECTION_SIZE = 416


def trainer_classes() -> dict[str, type]:
    """Every exported class whose name ends in `Trainer`, read off the package.

    Listed nowhere: a list of trainers is exactly the shape of thing `0.8.0a27` and `#300`
    both cost, and the reason this test exists is that somebody forgot one.
    """
    return {
        name: obj
        for name, obj in vars(training).items()
        if inspect.isclass(obj) and name.endswith("Trainer")
    }


def hook_names() -> list[str]:
    """Every hook the protocol declares, so a fifth is covered by arriving."""
    return sorted(
        name
        for name, value in vars(Callback).items()
        if name.startswith("on_") and inspect.isfunction(value)
    )


class Recorder(Callback):
    """Counts each hook rather than noting it, because `on_batch_end` firing once per epoch
    and once per step look the same to a set."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = dict.fromkeys(hook_names(), 0)
        self.steps: list[int] = []

    def on_train_begin(self, state: TrainingState) -> None:
        self.calls["on_train_begin"] += 1

    def on_batch_end(self, state: TrainingState) -> None:
        self.calls["on_batch_end"] += 1
        self.steps.append(state.step)

    def on_epoch_end(self, state: TrainingState) -> bool:
        self.calls["on_epoch_end"] += 1
        return False

    def on_train_end(self, state: TrainingState) -> None:
        self.calls["on_train_end"] += 1


class SegmentationStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(x.shape[0], 2, 8, 8) + self.weight


class DetectionStub(nn.Module):
    """Three grids of the shapes a YOLOv3 target has, driven by one real parameter so that
    an optimiser step actually happens."""

    def __init__(self, shapes) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.base = [torch.randn(1, *shape) for shape in shapes]

    def forward(self, x):
        return [base + self.weight for base in self.base]


def build_segmentation(batches: int = 4):
    spec = SegmentationModel(
        name="s", architecture="u_net", input_shape=(8, 8), blocks=1, filters=2, epochs=1
    )
    trainer = Trainer(SegmentationStub(), spec, device="cpu", callbacks=[])
    loader = [(torch.zeros(1, 3, 8, 8), torch.zeros(1, 2, 8, 8))] * batches
    return trainer, loader


def build_detection(batches: int = 4, accumulate: int = 1):
    encoded = encode(
        [[10, 10, 170, 170], [200, 40, 250, 90], [300, 300, 318, 318]],
        [0, 1, 2],
        anchors=ANCHORS,
        input_shape=(DETECTION_SIZE, DETECTION_SIZE),
        n_class=3,
    )
    targets = [torch.from_numpy(target)[None] for target in encoded.targets]
    spec = DetectionModel(name="d", input_shape=(DETECTION_SIZE, DETECTION_SIZE), epochs=1)
    trainer = DetectionTrainer(
        DetectionStub([tuple(t.shape[1:]) for t in targets]),
        spec,
        anchors=ANCHORS,
        n_class=3,
        device="cpu",
        callbacks=[],
        accumulate=accumulate,
    )
    loader = [(torch.zeros(1, 3, DETECTION_SIZE, DETECTION_SIZE), targets)] * batches
    return trainer, loader


class ClassificationStub(nn.Module):
    def __init__(self, n_class: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.n_class = n_class

    def forward(self, x):
        return torch.arange(self.n_class, dtype=torch.float32).repeat(x.shape[0], 1) + self.weight


def build_classification(batches: int = 4):
    """Both classes in the pass, because the default metric is `auc` and a ranking needs
    something to rank against - which is the engine's reason for stratifying a split and
    is visible here as a two-line fixture."""
    # `blocks=2` because the spec refuses a side below 2**blocks and the default is 4 -
    # which is slice 2's validator catching this fixture rather than the fixture being odd.
    spec = ClassificationModel(name="c", input_shape=(8, 8), blocks=2, filters=2, epochs=1)
    trainer = ClassificationTrainer(
        ClassificationStub(), spec, n_class=2, device="cpu", callbacks=[]
    )
    loader = [
        (torch.zeros(2, 3, 8, 8), torch.tensor([0, 1], dtype=torch.long)) for _ in range(batches)
    ]
    return trainer, loader


#: How to build each trainer for one cheap epoch. Keyed by class name so the completeness
#: test below can compare it against what the package actually exports.
BUILDERS = {
    "Trainer": build_segmentation,
    "DetectionTrainer": build_detection,
    "ClassificationTrainer": build_classification,
}


# --------------------------------------------------------- the second contract: a freeze


class WithTransfer(nn.Module):
    """A model shaped like a pretrained one, without a network.

    Every trainer that can hold a transfer reads `model.encoder.transferred`, which is what
    `PretrainedEncoder` sets for the layers that arrived with somebody else's weights. A
    stub with that attribute exercises the freeze offline, and whether the weights are
    ImageNet's or random is irrelevant to whether they are held still.
    """

    def __init__(self, segmentation: bool) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.transferred = nn.Sequential(nn.BatchNorm2d(3))
        self.ours = nn.Parameter(torch.zeros(1))
        self.segmentation = segmentation

    def forward(self, x):
        # run the batch norm, so its statistics would move if it were in training mode
        self.encoder.transferred(x)
        if self.segmentation:
            return torch.zeros(x.shape[0], 2, 8, 8) + self.ours
        value = x.mean(dim=(1, 2, 3))
        return torch.stack([value, -value], dim=1) + 0.0 * self.ours


def build_segmentation_transfer():
    model = WithTransfer(segmentation=True)
    spec = SegmentationModel(
        name="s",
        architecture="u_net",
        input_shape=(8, 8),
        blocks=1,
        filters=2,
        epochs=3,
        encoder="resnet34",
        freeze_encoder=2,
        metrics=[],
    )
    trainer = Trainer(model, spec, device="cpu", callbacks=[])
    loader = [(torch.rand(2, 3, 8, 8) + 5.0, torch.zeros(2, 2, 8, 8))] * 2
    return trainer, model, loader


def build_classification_transfer():
    model = WithTransfer(segmentation=False)
    spec = ClassificationModel(
        name="c",
        input_shape=(32, 32),
        encoder="resnet34",
        freeze_encoder=2,
        epochs=3,
        metrics=[],
    )
    trainer = ClassificationTrainer(model, spec, n_class=2, device="cpu", callbacks=[])
    loader = [(torch.rand(2, 3, 32, 32) + 5.0, torch.tensor([0, 1], dtype=torch.long))] * 2
    return trainer, model, loader


#: The trainers that can hold a transfer. Detection is not one of them - YOLOv3 carries its
#: own backbone and `parameter_groups(self.model, None, None)` says so - which is why this
#: is a second mapping rather than the one above with a hole in it.
TRANSFERS = {
    "Trainer": build_segmentation_transfer,
    "ClassificationTrainer": build_classification_transfer,
}


def test_the_transfer_builders_cover_every_trainer_that_reads_one():
    """A trainer that reads `transferred` and is missing here is a trainer whose freeze
    nothing checks, which is exactly the state the segmentation trainer was in."""
    reads_a_transfer = {
        name for name, builder in BUILDERS.items() if hasattr(builder()[0], "transferred")
    }
    assert set(TRANSFERS) <= reads_a_transfer
    assert set(reads_a_transfer) - set(TRANSFERS) == {"DetectionTrainer"} or not (
        set(reads_a_transfer) - set(TRANSFERS)
    ), sorted(reads_a_transfer - set(TRANSFERS))


@pytest.mark.parametrize("name", sorted(TRANSFERS))
def test_a_frozen_transfer_stays_frozen_through_model_train(name):
    """`model.train(True)` reaches every submodule, so a transfer put in `eval()` at the
    top of the epoch is switched straight back unless the trainer re-applies it **after**
    that call. Both halves are asserted, and the second is the one with a consequence a
    reader can see: a frozen BatchNorm in training mode rewrites its running statistics
    from every batch, so the thing being protected changes while the weights do not.

    Measured: with the re-application in place the running mean stays at 0.000000 for the
    frozen epochs and moves to 3.137033 the epoch the transfer is released.
    """
    trainer, model, loader = TRANSFERS[name]()
    transferred = model.encoder.transferred
    assert trainer.transferred is transferred, "the trainer found the transfer"
    seen: list[tuple[int, bool, bool, float]] = []

    class Watch(Callback):
        def on_epoch_end(self, state) -> bool:
            seen.append(
                (
                    state.epoch,
                    all(not p.requires_grad for p in transferred.parameters()),
                    transferred.training,
                    round(transferred[0].running_mean.abs().sum().item(), 6),
                )
            )
            return False

    trainer.callbacks = [Watch()]
    trainer.fit(loader, epochs=3)

    frozen = [(epoch, grad_off, training) for epoch, grad_off, training, _ in seen]
    assert frozen == [(1, True, False), (2, True, False), (3, False, True)]
    # the statistics, which is what the eval() half is for
    assert [mean for *_, mean in seen[:2]] == [0.0, 0.0]
    assert seen[-1][-1] > 0.0


# ------------------------------------------------------------ the first contract: hooks


def test_the_discovery_found_the_trainers_and_the_hooks():
    """The instrument before the subject. Two empty sets compare equal, and a test that
    iterates over nothing passes - which is the form this whole file's failure takes."""
    found = trainer_classes()
    assert set(found) == {"Trainer", "DetectionTrainer", "ClassificationTrainer"}, found
    assert hook_names() == ["on_batch_end", "on_epoch_end", "on_train_begin", "on_train_end"]


def test_every_trainer_the_package_exports_has_a_builder_here():
    """A trainer this file cannot build is a trainer this file is not testing, and it would
    otherwise be invisible: the parametrised test below would simply have one fewer case.
    The classification trainer will fail this on the day it is exported, which is the point.
    """
    assert set(trainer_classes()) == set(BUILDERS), (
        f"these trainers have no builder in this file: "
        f"{sorted(set(trainer_classes()) - set(BUILDERS))}"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_trainer_calls_every_hook_the_protocol_declares(name):
    """Behavioural rather than a grep over the source: what matters is that the hook fires
    on a real pass, and a call that is there but unreachable reads identically to a missing
    one in any static reading."""
    trainer, loader = BUILDERS[name]()
    recorder = Recorder()
    trainer.callbacks = [recorder]
    trainer.fit(loader, epochs=1)

    missing = [hook for hook, count in recorder.calls.items() if count == 0]
    assert not missing, f"{name} never called: {', '.join(missing)}"
    assert recorder.calls["on_train_begin"] == 1
    assert recorder.calls["on_train_end"] == 1
    assert recorder.calls["on_epoch_end"] == 1
    assert recorder.calls["on_batch_end"] == len(loader)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_step_number_a_hook_sees_counts_up_from_one(name):
    """`state.step` is what a callback with a schedule on it reads, so a trainer that fires
    the hook without advancing it would pass the test above and still be wrong."""
    trainer, loader = BUILDERS[name]()
    recorder = Recorder()
    trainer.callbacks = [recorder]
    trainer.fit(loader, epochs=1)
    assert recorder.steps == list(range(1, len(loader) + 1))


def test_a_detection_hook_fires_per_step_rather_than_per_batch():
    """Only a distinction when `accumulate` is above 1, and there it decides what `decay`
    means. Measured on a weight that moves by 1.0 per step, decay 0.9, 32 batches: at
    `accumulate: 1` the two readings are identical to the digit, and at 4 they are 3.3047
    against 5.4728 with the true weight at 8.0 - so folding per batch would shrink an
    exponential average's horizon by exactly the accumulation.
    """
    trainer, loader = build_detection(batches=8, accumulate=4)
    recorder = Recorder()
    trainer.callbacks = [recorder]
    trainer.fit(loader, epochs=1)
    assert recorder.calls["on_batch_end"] == 2, "8 batches at accumulate 4 is two steps"
    assert recorder.steps == [1, 2]


def test_the_group_left_over_at_the_end_of_an_epoch_is_a_step_too():
    """Five batches at accumulate 2 is two full groups and one left over, and the trailing
    group is stepped rather than discarded - so it is a step the hook has to see.

    No test here reached that branch until this one: every loader was a whole number of
    groups, so silencing the trailing call changed nothing and a mutation survived. §4's
    fixture that never contained the case, in the file written to stop exactly that.
    """
    trainer, loader = build_detection(batches=5, accumulate=2)
    recorder = Recorder()
    trainer.callbacks = [recorder]
    trainer.fit(loader, epochs=1)
    assert recorder.calls["on_batch_end"] == 3, "two full groups and the remainder"
    assert recorder.steps == [1, 2, 3]


def test_a_validation_pass_is_not_a_step():
    """A batch the optimiser did not step on must not fire the hook, or an average would
    fold weights that had not moved - and the validation pass would be counted as training."""
    trainer, loader = build_segmentation(batches=3)
    recorder = Recorder()
    trainer.callbacks = [recorder]
    trainer.fit(loader, loader, epochs=1)
    assert recorder.calls["on_batch_end"] == 3, "the training pass only"


def test_the_callback_written_for_the_fourth_hook_now_averages_on_a_detector():
    """The defect itself, end to end, with the segmentation trainer as the control - which
    is what makes 8 a finding rather than a number."""
    from pyplatypus.training.callbacks import Ema

    detection, detection_loader = build_detection(batches=4)
    folding = Ema(decay=0.9)
    detection.callbacks = [folding]
    history = detection.fit(detection_loader, epochs=2)
    assert folding._folded == 8
    assert history.records[-1]["ema_steps"] == 8.0

    segmentation, segmentation_loader = build_segmentation(batches=4)
    control = Ema(decay=0.9)
    segmentation.callbacks = [control]
    control_history = segmentation.fit(segmentation_loader, epochs=2)
    assert control._folded == folding._folded
    assert control_history.records[-1]["ema_steps"] == history.records[-1]["ema_steps"]
