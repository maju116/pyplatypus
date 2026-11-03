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

One subtlety used to decide the fixtures below, and it was a defect rather than a subtlety.
This docstring said: *the object count is `clamp(min=1)`, so a batch with no object in some
grid contributes a denominator of 1 rather than 0 - and then regrouping can move the total.
Exact invariance therefore holds when every batch has something in every grid, which is what
`three_sizes` is for.* So the limitation was written down and designed around: `three_sizes`
gives every image a box at every scale **so no grid is ever empty**, and the invariance test
below could not see the one case where invariance failed.

`pyplatypus#352` removed it. The clamp protected nothing - `count` is a divisor only inside
`if has_object.any()`, where the sum is at least 1 by construction - so the recorded count is
the true one now and an empty grid contributes nothing to either side of the pooling.
`test_an_epoch_is_the_same_when_a_grid_is_empty_in_some_batch` is the case the old fixture
avoided, and `three_sizes` stays because a fixture where two denominators are proportional
cannot tell which one the code used.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from pyplatypus.detection.encode import encode
from pyplatypus.detection.loss import LossParts, Yolo3Loss
from pyplatypus.spec.detection import DetectionModel
from pyplatypus.training.detection_trainer import TERMS, DetectionTrainer


@dataclass(frozen=True)
class Image:
    """One fixture image: its targets per grid, and the boxes they were encoded from."""

    targets: list[torch.Tensor]
    boxes: torch.Tensor


ANCHORS = (((0.25, 0.25), (0.15, 0.30)), ((0.10, 0.10), (0.06, 0.12)), ((0.04, 0.04), (0.02, 0.05)))
SIZE = 416


def encoded_image(boxes, labels) -> Image:
    """One image's targets AND the boxes they came from, which is what a loader carries.

    The boxes travel beside the targets since `pyplatypus#359`: the ignore mask has to be
    built from the image's own boxes, because `encode` drops one whose cell and slot are
    taken. So these fixtures carry both, and the stub loader below hands over the same
    three things `make_detection_loader` does.
    """
    encoded = encode(boxes, labels, anchors=ANCHORS, input_shape=(SIZE, SIZE), n_class=3)
    return Image(
        targets=[torch.from_numpy(t)[None] for t in encoded.targets],
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    )


def three_sizes(offset: int) -> Image:
    """One image, with a box for each scale so no grid is ever empty."""
    return encoded_image(
        [
            [offset + 10, offset + 10, offset + 170, offset + 170],
            [offset + 200, offset + 40, offset + 250, offset + 90],
            [offset + 300, offset + 300, offset + 318, offset + 318],
        ],
        [0, 1, 2],
    )


def crowded(offset: int) -> Image:
    """An image with the same three scales covered but MORE objects in the smallest one.

    The point is the ratio between the two denominators. `three_sizes` gives every image
    three objects and a fixed number of cells, so weighting by objects and weighting by
    cells come out proportional - and a fixture where two quantities are proportional
    cannot tell which one the code used. Mutating `no_object`'s denominator from cells to
    objects passed every test here until this existed.

    Still one object per grid at least, so the `clamp(min=1)` never binds and exact
    invariance survives.
    """
    return encoded_image(
        [
            [offset + 10, offset + 10, offset + 170, offset + 170],
            [offset + 200, offset + 40, offset + 250, offset + 90],
            [offset + 300, offset + 300, offset + 318, offset + 318],
            [offset + 260, offset + 300, offset + 278, offset + 318],
            [offset + 300, offset + 260, offset + 318, offset + 278],
        ],
        [0, 1, 2, 0, 1],
    )


def batched(images: list[Image]):
    """Several images stacked into one batch, exactly as `_collate_detection` hands it over.

    Three things, not two: the boxes are a LIST because their count varies per image, which
    is what the real collate does and why it cannot stack them. `pyplatypus#359`.
    """
    grids = len(images[0].targets)
    targets = [torch.cat([one.targets[grid] for one in images], dim=0) for grid in range(grids)]
    boxes = [one.boxes for one in images]
    return torch.zeros(len(images), 3, SIZE, SIZE), targets, boxes


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


def epoch_over(groups: list[list[Image]], *, prefix: str = "val", plant=None) -> dict[str, float]:
    """Run one epoch over images grouped into the given batches.

    `prefix` is what the keys are named with, and it defaults to the validation pass because
    that is what every test here but one asks about. `train=False` either way: this builds
    an epoch's figures from fixed replies and never steps an optimiser.

    `plant` is a box to make one empty cell decode to, for the two tests about where the
    loss gets its truths: without it every confident prediction sits in a cell the target
    already fills, and those are excluded from `no_object` whichever route built the mask.
    """
    loader = []
    replies = []
    first = 0
    for group in groups:
        images, targets, boxes = batched(group)
        reply = predict_for(targets, first)
        if plant is not None:
            reply = planted(reply, plant)
        replies.append(reply)
        loader.append((images, targets, boxes))
        first += len(group)

    return DetectionTrainer._run_epoch(Stub(replies), loader, train=False, prefix=prefix)


def with_a_collision(offset: int) -> Image:
    """Two boxes of one size whose centres fall in ONE cell of the fine grid.

    The fine grid's stride is 8, so two 18-pixel boxes whose centres are in one 8-pixel cell
    want the same cell and the same anchor: `encode` keeps the first and counts the second
    `unplaced`. This is the only fixture in this file on which the two sources of truth
    disagree, and therefore the only one that can see whether the loss was handed the
    image's boxes. `pyplatypus#359`.

    **The offset of 5 pixels is measured rather than chosen, and it is the finding's own
    limit.** A collision needs both centres in one cell, so the two boxes are close - and if
    they are too close, a prediction of the one that did not fit has a high IoU with the one
    that did, which the ignore mask then covers either way. Over the eight offsets that
    keep the collision here:

        offset  2   IoU 0.653   both routes ignore it
        offset  3   IoU 0.532   both routes ignore it
        offset  4   IoU 0.434   only the boxes route ignores it
        offset  5   IoU 0.353   only the boxes route ignores it

    So `#359` bites in a window rather than on any crowded dataset: the two boxes have to sit
    far enough apart inside one cell for their overlap to fall under `ignore_threshold`.
    5 is the widest offset this cell allows.
    """
    return encoded_image(
        [
            [offset + 10, offset + 10, offset + 170, offset + 170],
            [offset + 200, offset + 40, offset + 250, offset + 90],
            [offset + 300, offset + 300, offset + 318, offset + 318],
            [offset + 295, offset + 295, offset + 313, offset + 313],
        ],
        [0, 1, 2, 2],
    )


def planted(prediction, box, *, grid: int = 0, anchors=None):
    """A prediction with one empty cell made to decode EXACTLY to `box`.

    The ignore mask only has anything to say about a cell with no object in it, and
    `predict_for` builds its predictions from the targets - so every confident box it
    produces sits where the target already holds one, and those cells are excluded from
    `no_object` either way. Without planting one, a fixture with an unplaced truth reads the
    same through both routes and a test over it asserts nothing. `pyplatypus#359`.

    Cell (0, 0) of the coarse grid, which none of these fixtures occupies.
    """
    anchors = anchors or ANCHORS
    planted_prediction = [one.clone() for one in prediction]
    rows, columns = planted_prediction[grid].shape[1:3]
    centre_x = (box[0] + box[2]) / 2 / SIZE
    centre_y = (box[1] + box[3]) / 2 / SIZE
    width = (box[2] - box[0]) / SIZE
    height = (box[3] - box[1]) / SIZE

    # THE CELL THAT CONTAINS THE CENTRE, not cell (0, 0). `decode` reads the first two
    # numbers through a sigmoid as an offset WITHIN the cell, so a box whose centre lies
    # elsewhere cannot be expressed from here at all: the first version of this helper wrote
    # into (0, 0) and the planted prediction decoded to a box near the origin, 0.077 of the
    # frame instead of 0.748. Both routes then ignored nothing and the test it was written
    # for asserted that two identical numbers were identical.
    column = min(int(centre_x * columns), columns - 1)
    row = min(int(centre_y * rows), rows - 1)
    assert not bool(planted_prediction[grid][0, row, column, 0, 4] > 50), (
        "planting into a cell the fixture already fills; the mask excludes those either way"
    )

    cell = planted_prediction[grid][0, row, column, 0]
    cell[0] = torch.logit(torch.tensor(centre_x * columns - column).clamp(1e-6, 1 - 1e-6))
    cell[1] = torch.logit(torch.tensor(centre_y * rows - row).clamp(1e-6, 1 - 1e-6))
    cell[2] = float(np.log(width / anchors[grid][0][0]))
    cell[3] = float(np.log(height / anchors[grid][0][1]))
    # CONFIDENT, which is the whole point: `no_object` is a mean over the punished cells,
    # and `predict_for` gives every empty cell in an image the same objectness logit - so
    # dropping one of them from that mean changes nothing. Measured: with the objectness
    # left alone, the two routes punished 337 cells and 336 and returned `no_object`
    # identical to the bit. A prediction the mask is meant to protect is a prediction that
    # says something is there.
    cell[4] = 5.0
    return planted_prediction


def test_the_loss_is_given_the_images_boxes_rather_than_the_targets():
    """`pyplatypus#359` at the level that matters: the route, not the helper.

    The first version of this guard called `_normalised_boxes` and `_ignore_mask` directly,
    so it asserted that the helper works and said nothing about whether `forward` or the
    trainer use it. Measured: reverting `forward`'s truth selection left the whole file
    **passing**, and so did stopping the trainer from forwarding the boxes. A guard one
    level below the change is a guard the change can be removed under.

    So this compares the two routes through `forward` itself, on a fixture where they
    differ - a box the target could not hold - with an uncrowded fixture as the control.
    """
    collided = [with_a_collision(0)]
    _, targets, boxes = batched(collided)
    unplaced_box = [295, 295, 313, 313]
    prediction = planted(predict_for(targets, 0), unplaced_box)
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3)

    with_boxes = loss(prediction, targets, boxes).as_dict()
    from_targets = loss(prediction, targets).as_dict()

    # The unplaced object is real: without it the two routes see the same truths and this
    # test asserts nothing.
    assert len(boxes[0]) == 4
    assert with_boxes["no_object"] != from_targets["no_object"], (
        "the two routes agreed on a fixture built to make them differ, so either the "
        "collision stopped happening or the boxes are not reaching the loss"
    )

    # And the control: with nothing unplaced, the two routes are the same answer.
    plain = [three_sizes(0)]
    _, plain_targets, plain_boxes = batched(plain)
    plain_prediction = predict_for(plain_targets, 0)
    assert loss(plain_prediction, plain_targets, plain_boxes).as_dict() == pytest.approx(
        loss(plain_prediction, plain_targets).as_dict(), rel=1e-6
    )


def test_the_trainer_hands_the_boxes_to_the_loss():
    """The third level: the epoch, which is where `#359`'s fix is actually wired.

    An epoch over the colliding fixture has to reproduce `forward`'s answer **with** the
    boxes rather than without them. Reverting the trainer's forward left every other test
    in this file green, this one included until it existed.
    """
    collided = [with_a_collision(0)]
    _, targets, boxes = batched(collided)
    unplaced_box = [295, 295, 313, 313]
    prediction = planted(predict_for(targets, 0), unplaced_box)
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3)

    epoch = epoch_over([collided], plant=unplaced_box)
    with_boxes = loss(prediction, targets, boxes).as_dict()
    from_targets = loss(prediction, targets).as_dict()

    assert epoch["val_no_object"] == pytest.approx(with_boxes["no_object"], rel=1e-6)
    # The control that the two differ at all, so the assertion above has a wrong answer to
    # fail against.
    assert with_boxes["no_object"] != from_targets["no_object"]


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
            _, targets, _ = batched(group)
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
    _, targets, boxes = batched(images)
    prediction = predict_for(targets, 0)

    # The boxes go to the direct call too, because the epoch passes them: comparing the
    # boxes route against the targets fallback would be comparing two computations rather
    # than asserting the pooling's shape. `pyplatypus#359`.
    direct = Yolo3Loss(anchors=ANCHORS, n_class=3)(prediction, targets, boxes).as_dict()
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


def one_scale(offset: int, side: int) -> Image:
    """One image with a single box, so it occupies ONE grid and leaves the other two empty.

    The fixture `three_sizes` was written to avoid; `pyplatypus#352` is what it was avoiding.
    """
    return encoded_image([[offset + 10, offset + 10, offset + 10 + side, offset + 10 + side]], [0])


def occupied_grids(image: Image) -> list[int]:
    """Which grids this image actually has an object in, counted the way the loss counts."""
    return [
        index for index, target in enumerate(image.targets) if bool((target[..., 4] > 0.5).any())
    ]


def test_an_epoch_is_the_same_when_a_grid_is_empty_in_some_batch():
    """The case the invariance above was calibrated to avoid. `pyplatypus#352`.

    A batch with no object at some scale used to contribute 0 to that grid's numerator and
    **1** to its denominator, because the recorded object count was `clamp(min=1)`. So every
    reported term was deflated by `objects / (objects + empty batch-grids)` and the epoch
    became a function of `batch_size` - the exact property these tests exist to deny, on the
    exact data their fixture excluded.

    Measured through `fit()` on a frozen model before the fix, two images each occupying one
    scale: `val_loss` 6.58010 at `batch_size 1` against 11.04327 at 2. With a box at every
    scale in every image the same probe gave 16.46951 against 16.46593, which is the control
    that said this was the clamp and not the batching.
    """
    coarse = one_scale(0, 160)
    fine = one_scale(40, 18)

    # THE CONTROL THAT THE CASE IS PRESENT. Without this the test passes on any fixture
    # whose grids all happen to be occupied, which is how the invariance test above stayed
    # green through the defect for as long as it existed.
    assert occupied_grids(coarse) != occupied_grids(fine), (
        "both images occupy the same grids, so no batching leaves one empty and this test "
        f"is asking nothing: {occupied_grids(coarse)} and {occupied_grids(fine)}"
    )

    apart = epoch_over([[coarse], [fine]])
    together = epoch_over([[coarse, fine]])

    for key in apart:
        assert apart[key] == pytest.approx(together[key], rel=1e-6), (
            f"{key} moved with the batching on an empty grid: {apart[key]} against {together[key]}"
        )


def test_what_an_epoch_writes_is_what_a_callback_may_watch():
    """§3F's pair for `pyplatypus#358`: one term list, two readers.

    `DetectionTrainer._run_epoch` writes `f"{prefix}_{term}"` for every term and
    `f"{prefix}_loss"`; `DetectionModel.monitorable` says which names a callback may watch.
    Those are two answers to one question, and before `#358` they disagreed in both
    directions at once: the set offered `train_foreground` and `val_foreground`, which this
    module never writes, and omitted the four terms, which it always does. A callback
    watching the first was accepted and killed the run with a `KeyError` at the end of its
    first epoch; a user wanting to watch the second was told it was unavailable.

    Asserted over the epoch this file already builds from a stub rather than over a trained
    run, because the question is which keys exist and not what they are worth.
    """
    model = DetectionModel(
        name="d",
        input_shape=(128, 128),
        epochs=1,
        batch_size=2,
        anchors_per_grid=2,
        optimizer={"name": "adam", "learning_rate": 1e-3},
    )
    watchable = model.monitorable

    # Both controls first: two empty sets compare equal, which is the form this failure
    # takes when the extraction breaks rather than the code.
    assert len(watchable) >= 8, f"monitorable is implausibly small: {sorted(watchable)}"

    for prefix in ("train", "val"):
        epoch = epoch_over([[three_sizes(0), crowded(8)]], prefix=prefix)
        written = set(epoch)
        assert written, f"the {prefix} epoch wrote nothing"
        offered = {name for name in watchable if name.startswith(f"{prefix}_")}

        assert written == offered, (
            f"what a {prefix} epoch writes and what a callback may watch disagree: "
            f"written only {sorted(written - offered)}, offered only {sorted(offered - written)}"
        )

    # And the name that cost `#358`: a segmentation epoch reports it, a detection one does
    # not, so it must not be on offer here.
    assert "val_foreground" not in watchable


def test_the_term_list_is_what_the_loss_actually_breaks_out():
    """The third reader, and the one the test above cannot be.

    `DETECTION_TERMS` has two readers - the epoch's keys and `monitorable` - so the pair
    above compares them against each other and **both move together when the list itself
    changes**. Measured: dropping `"classes"` from the tuple leaves that test passing,
    because the epoch then stops writing it and the set stops offering it.

    What pins the list is the thing it is a list *of*: `LossParts`' fields, minus `total`,
    which is their weighted sum rather than a term. Written rather than left as a limit in
    a docstring, because §4's newest category is that a limitation explained in prose and
    designed around is a defect with an alibi - and this one closes in two lines.
    """
    from dataclasses import fields

    import torch as _torch

    # The terms are LossParts' SCALAR fields: `total` is their weighted sum rather than one
    # of them, and `grids` is the per-scale breakdown each of them was pooled from. Picked
    # by annotation rather than by name, so a sixth field added later lands in this test
    # instead of being skipped by a list of exceptions somebody has to remember.
    declared = tuple(
        f.name for f in fields(LossParts) if f.type is _torch.Tensor or f.type == "torch.Tensor"
    )
    assert declared, "no scalar fields found on LossParts, so this test is asking nothing"
    assert declared[0] == "total", f"expected `total` first, got {declared}"
    assert TERMS == declared[1:], (
        f"the term list and what the loss breaks out disagree: {TERMS} against {declared[1:]}"
    )
