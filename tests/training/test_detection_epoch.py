"""What a detection epoch reports, and what it refuses to depend on. `maju116/rplatypus#218`.

`DetectionTrainer._run_epoch` used to end in `{key: value / batches}`, exactly as the
segmentation trainer did before `#196`. The thing being averaged is different, and that is
why the same repair does not transfer: every component of the YOLO loss is divided inside
the loss, three of them by a grid's **object count** and the fourth by the **cells it
supervises**. Averaging those over batches weights a frame of thirty blood cells the same as
a frame of two.

So the epoch pools by what the loss divided by, and the property that buys is the one these
tests assert: **the same examples report the same epoch however they are grouped into
batches.**

One subtlety decides the fixtures below. The object count is `clamp(min=1)`, so a batch with
no object in some grid contributes a denominator of 1 rather than 0 - and then regrouping
*can* move the total. Exact invariance therefore holds when every batch has something in
every grid, which is what `three_sizes` is for: one small, one medium and one large box per
image, so each lands in a different scale.
"""

import pytest
import torch

from pyplatypus.detection.encode import encode
from pyplatypus.detection.loss import Yolo3Loss
from pyplatypus.training.detection_trainer import TERMS, DetectionTrainer

ANCHORS = (((0.25, 0.25), (0.15, 0.30)), ((0.10, 0.10), (0.06, 0.12)), ((0.04, 0.04), (0.02, 0.05)))
SIZE = 416


def three_sizes(offset: int) -> list[torch.Tensor]:
    """One image's targets, with a box for each scale so no grid is ever empty."""
    boxes = [
        [offset + 10, offset + 10, offset + 170, offset + 170],
        [offset + 200, offset + 40, offset + 250, offset + 90],
        [offset + 300, offset + 300, offset + 318, offset + 318],
    ]
    encoded = encode(boxes, [0, 1, 2], anchors=ANCHORS, input_shape=(SIZE, SIZE), n_class=3)
    return [torch.from_numpy(t)[None] for t in encoded.targets]


def crowded(offset: int) -> list[torch.Tensor]:
    """An image with the same three scales covered but MORE objects in the smallest one.

    The point is the ratio between the two denominators. `three_sizes` gives every image
    three objects and a fixed number of cells, so weighting by objects and weighting by
    cells come out proportional - and a fixture where two quantities are proportional
    cannot tell which one the code used. Mutating `no_object`'s denominator from cells to
    objects passed every test here until this existed.

    Still one object per grid at least, so the `clamp(min=1)` never binds and exact
    invariance survives.
    """
    boxes = [
        [offset + 10, offset + 10, offset + 170, offset + 170],
        [offset + 200, offset + 40, offset + 250, offset + 90],
        [offset + 300, offset + 300, offset + 318, offset + 318],
        [offset + 260, offset + 300, offset + 278, offset + 318],
        [offset + 300, offset + 260, offset + 318, offset + 278],
    ]
    encoded = encode(boxes, [0, 1, 2, 0, 1], anchors=ANCHORS, input_shape=(SIZE, SIZE), n_class=3)
    return [torch.from_numpy(t)[None] for t in encoded.targets]


def batched(images: list[list[torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Several images' targets stacked into one batch, as a loader would hand them over."""
    grids = len(images[0])
    targets = [torch.cat([one[grid] for one in images], dim=0) for grid in range(grids)]
    return torch.zeros(len(images), 3, SIZE, SIZE), targets


class InOrder:
    """A model that replies from a list, one call at a time.

    Keyed by the ORDER of the call rather than by the batch's shape, which is the version
    this started as and which quietly broke the comparison: two batches of the same size
    collided in the table, so a 3+3 split scored both halves against one prediction while a
    4+2 split scored each against its own. The test then compared two different things and
    reported the difference as a property of the pooling.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def train(self, flag):
        pass

    def __call__(self, images):
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


class Stub:
    """Just enough trainer to run an epoch: a model that answers in order, and the real
    loss. Built rather than trained, because what is under test is the arithmetic of the
    epoch and not anything a model learns."""

    def __init__(self, replies):
        self.model = InOrder(replies)
        self.loss_fn = Yolo3Loss(anchors=ANCHORS, n_class=3)
        self.device = "cpu"
        self.accumulate = 1


def predict_for(targets: list[torch.Tensor], first: int) -> list[torch.Tensor]:
    """A wrong but finite prediction that differs PER IMAGE, not just per batch.

    `t * 0.5 - 0.25` alone was the first version and it hides a whole class of error: on an
    empty cell the target is zero, so every image gets the same objectness logit and
    `no_object` comes out near enough identical for every batch. A term that does not vary
    between images cannot show which denominator was used to weight it, and mutating
    `no_object`'s divisor from cells to objects passed every test here until this varied.

    `first` is the index of the group's first image, so the bias follows an image rather
    than its position within a batch - otherwise regrouping would change the prediction and
    the invariance tests would be comparing different runs.
    """
    out = []
    for target in targets:
        prediction = target * 0.5 - 0.25
        bias = torch.arange(target.shape[0], dtype=prediction.dtype) + first
        shape = [target.shape[0]] + [1] * (prediction.dim() - 2)
        prediction[..., 4] = prediction[..., 4] + bias.reshape(shape) * 0.3
        out.append(prediction)
    return out


def epoch_over(groups: list[list[list[torch.Tensor]]]) -> dict[str, float]:
    """Run one validation epoch over images grouped into the given batches."""
    loader = []
    replies = []
    first = 0
    for group in groups:
        images, targets = batched(group)
        replies.append(predict_for(targets, first))
        loader.append((images, targets))
        first += len(group)

    return DetectionTrainer._run_epoch(Stub(replies), loader, train=False, prefix="val")


def test_an_epoch_is_the_same_however_the_examples_are_batched():
    """Six images, grouped 3+3 and then 2+2+2 and then 4+2 - the same epoch every time.

    This is the property the fix buys and the one the old arithmetic did not have: a mean
    over batches makes the reported number a function of `batch_size`, which is a training
    setting with no business in a reported score.
    """
    # A MIXED set: three images with three objects and three with five, so the number of
    # objects per image varies while the number of cells per image does not. Without that,
    # the two denominators are proportional and this test cannot see which one was used -
    # which was true of it until `crowded` was written.
    images = [
        three_sizes(0),
        crowded(8),
        three_sizes(16),
        crowded(24),
        three_sizes(32),
        crowded(40),
    ]

    even = epoch_over([images[:3], images[3:]])
    pairs = epoch_over([images[0:2], images[2:4], images[4:6]])
    uneven = epoch_over([images[:4], images[4:]])

    # The bound is calibrated against both ends rather than loosened until green, which is
    # the rule CLAUDE.md §4 states for exactly this: float32 reductions over different
    # groupings add in a different order, and four batchings of this epoch differ by at
    # most 6.9e-08 relative. The old pooling moves the same terms by 1.94e-03. So 1e-6 sits
    # 14x above the noise and 1945x below the thing this test exists to catch.
    for key in even:
        assert even[key] == pytest.approx(pairs[key], rel=1e-6), f"{key} moved with the batching"
        assert even[key] == pytest.approx(uneven[key], rel=1e-6), f"{key} moved with the batching"


def test_the_old_pooling_would_have_moved_and_this_fixture_can_see_it():
    """The control, without which the test above passes on any arithmetic at all.

    If a mean over batches gave the same answer on this fixture, the assertion above would
    hold whatever `_run_epoch` did - so what has to be shown is that the two poolings
    genuinely differ here, and by how much.
    """
    images = [
        three_sizes(0),
        crowded(8),
        three_sizes(16),
        crowded(24),
        three_sizes(32),
        crowded(40),
    ]

    def mean_over_batches(groups):
        totals = {}
        count = 0
        first = 0
        loss = Yolo3Loss(anchors=ANCHORS, n_class=3)
        for group in groups:
            _, targets = batched(group)
            parts = loss(predict_for(targets, first), targets).as_dict()
            first += len(group)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        return {key: value / count for key, value in totals.items()}

    even = mean_over_batches([images[:3], images[3:]])
    uneven = mean_over_batches([images[:4], images[4:]])

    moved = max(abs(even[term] - uneven[term]) for term in TERMS)
    assert moved > 1e-6, (
        "the old pooling gives the same answer on this fixture, so the test above cannot "
        "tell the two apart and proves nothing"
    )


def test_the_epoch_reconstructs_its_total_rather_than_pooling_it():
    """`total` is a weighted sum of four terms divided by two different denominators, so it
    has no denominator of its own. Pooling it directly would be dividing by a number that
    does not describe it; it is rebuilt from the four, with the loss's own weights."""
    images = [three_sizes(offset) for offset in (0, 8, 16)]
    reported = epoch_over([images[:2], images[2:]])

    weights = Yolo3Loss(anchors=ANCHORS, n_class=3).weights
    rebuilt = sum(weights[term] * reported[f"val_{term}"] for term in TERMS)
    assert reported["val_loss"] == pytest.approx(rebuilt, rel=1e-12)


def test_one_batch_reproduces_the_loss_exactly():
    """The invariant that pins the shape of the pooling rather than its weighting.

    Over a single batch every denominator divides itself out: `sum_g (part_g * d_g / d_g)`
    is `sum_g part_g`, which is what `forward` already returned. So a one-batch epoch has to
    reproduce the loss's own four numbers to the bit.

    It is here because the invariance test above cannot see two of the three ways this can
    be wrong. Mixing the grids instead of summing them, and dividing `no_object` by the
    object count instead of by the cells it supervises, are **both still invariant to
    batching** - so a test calibrated on invariance alone passes on either. That is §4's
    check admitting the failure it was written to prevent, found by mutating the trainer
    three ways and watching only one of them fail.
    """
    images = [three_sizes(offset) for offset in (0, 8, 16)]
    _, targets = batched(images)
    prediction = predict_for(targets, 0)

    direct = Yolo3Loss(anchors=ANCHORS, n_class=3)(prediction, targets).as_dict()
    epoch = epoch_over([images])

    # Calibrated at both ends again: `GridParts` holds floats read off float32 tensors, so
    # `part * d / d` does not return the same bits - 2.58e-08 relative at worst here. A
    # collapsed grid moves the same numbers by about 67%. 1e-6 is 39x the round-trip and
    # five orders of magnitude below the failure.
    for term in TERMS:
        assert epoch[f"val_{term}"] == pytest.approx(direct[term], rel=1e-6), (
            f"a one-batch epoch reports {term} as {epoch[f'val_{term}']} where the loss "
            f"computed {direct[term]}"
        )
    assert epoch["val_loss"] == pytest.approx(direct["loss"], rel=1e-6)
