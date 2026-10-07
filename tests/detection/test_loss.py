"""The objective, and the two things about it that read like bugs and are not.

A perfect prediction does **not** give a loss of zero, and the run on BCCD plateaus at a
number far above zero. Both are properties of binary cross-entropy against a soft target,
whose minimum is the target's own entropy rather than nothing - so the tests below assert
the two things that actually characterise a correct objective: the gradient vanishes at the
optimum, and the loss falls monotonically towards it.
"""

import pytest
import torch
from torch.nn import functional as F

from pyplatypus.detection.encode import encode
from pyplatypus.detection.loss import Yolo3Loss
from pyplatypus.detection.metrics import DetectionError

ANCHORS = (((0.25, 0.25), (0.15, 0.30)), ((0.10, 0.10), (0.06, 0.12)), ((0.04, 0.04), (0.02, 0.05)))


def targets_for(boxes, labels, *, n_class=3, size=416, anchors=ANCHORS):
    encoded = encode(boxes, labels, anchors=anchors, input_shape=(size, size), n_class=n_class)
    return [torch.from_numpy(t)[None] for t in encoded.targets]


def perfect(targets, big=12.0):
    """The logits a model would have to emit to reproduce these targets exactly.

    `big` rather than infinity, which is the whole reason the encoder stores the offset
    instead of its logit: the old implementation's target was `logit(offset)` and went to
    `-inf` whenever a centre landed on a cell boundary.
    """
    out = []
    for target in targets:
        prediction = torch.zeros_like(target)
        has_object = target[..., 4] > 0.5
        offsets = target[..., 0:2].clamp(1e-6, 1 - 1e-6)
        prediction[..., 0:2] = torch.log(offsets / (1 - offsets))
        prediction[..., 2:4] = target[..., 2:4]
        prediction[..., 4] = torch.where(has_object, big, -big)
        prediction[..., 5:] = torch.where(target[..., 5:] > 0.5, big, -big)
        out.append(prediction)
    return out


# --- what "optimal" means here ----------------------------------------------------------


def test_the_gradient_vanishes_at_the_optimum():
    """Which is the property that matters for training, and the one to assert - not a loss
    of zero, which cross-entropy against a soft target cannot reach."""
    targets = targets_for([[40, 40, 140, 140], [250, 100, 290, 150]], [0, 2])
    ideal = [p.requires_grad_(True) for p in perfect(targets)]
    Yolo3Loss(anchors=ANCHORS, n_class=3)(ideal, targets).total.backward()

    largest = max(p.grad.abs().max().item() for p in ideal)
    assert largest < 1e-4


def test_the_floor_is_the_targets_own_entropy():
    """Stated as arithmetic so "the loss stopped at 5.9" can be recognised as convergence
    rather than as a stall. BCE between a logit and a soft target p bottoms out at
    -p ln p - (1-p) ln(1-p)."""
    for value in (0.25, 0.5, 0.75):
        target = torch.tensor([value])
        at_optimum = F.binary_cross_entropy_with_logits(torch.logit(target), target).item()
        entropy = -(
            value * torch.log(torch.tensor(value))
            + (1 - value) * torch.log(torch.tensor(1 - value))
        ).item()
        assert at_optimum == pytest.approx(entropy, abs=1e-6)


def test_three_of_the_four_terms_do_reach_zero():
    """Objectness, no-object and classes have hard 0/1 targets, so their floor is nothing.
    Only the coordinate term has a soft target, which is where the residual comes from."""
    targets = targets_for([[40, 40, 140, 140]], [1])
    parts = Yolo3Loss(anchors=ANCHORS, n_class=3)(perfect(targets), targets).as_dict()

    assert parts["objectness"] == pytest.approx(0.0, abs=1e-4)
    assert parts["no_object"] == pytest.approx(0.0, abs=1e-4)
    assert parts["classes"] == pytest.approx(0.0, abs=1e-4)
    assert parts["coordinates"] > 0.5


def test_the_loss_falls_as_a_prediction_approaches_the_target():
    targets = targets_for([[40, 40, 140, 140], [250, 100, 290, 150]], [0, 2])
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3)
    ideal = [p.detach() for p in perfect(targets)]

    scores = []
    for noise in (3.0, 2.0, 1.0, 0.5, 0.1, 0.0):
        torch.manual_seed(0)
        noisy = [p + torch.randn_like(p) * noise for p in ideal]
        scores.append(loss(noisy, targets).total.item())
    assert scores == sorted(scores, reverse=True), scores


# --- the four terms are on a comparable scale -------------------------------------------


def test_the_no_object_term_does_not_swamp_the_others():
    """The first version divided this by the object count while summing it over thousands
    of empty cells, so one object gave a no-object loss of 4915 against a coordinate loss
    of 5.5 - at which ratio the only thing a model can learn is to answer "nothing here".
    """
    targets = targets_for([[40, 40, 140, 140]], [0])
    zeros = [torch.zeros_like(t) for t in targets]
    parts = Yolo3Loss(anchors=ANCHORS, n_class=3)(zeros, targets).as_dict()

    assert parts["no_object"] < 10 * parts["coordinates"]
    # BCE(logit 0, target 0) is ln 2 per cell, and the loss sums the three grids, so the
    # floor for an all-zero prediction is 3 ln 2. Written out because the first version of
    # this expectation was ln 2 - the arithmetic forgot the sum over grids.
    assert parts["no_object"] == pytest.approx(3 * 0.6931, abs=0.01)


def test_the_parts_are_reported_separately():
    """One number cannot say what is wrong: a run whose coordinate loss falls while its
    objectness does not is finding the right places and refusing to commit."""
    targets = targets_for([[40, 40, 140, 140]], [0])
    parts = Yolo3Loss(anchors=ANCHORS, n_class=3)([torch.zeros_like(t) for t in targets], targets)
    assert set(parts.as_dict()) == {"loss", "coordinates", "objectness", "no_object", "classes"}
    assert parts.total == pytest.approx(
        parts.coordinates + parts.objectness + parts.no_object + parts.classes
    )


def test_small_boxes_are_weighted_up():
    """`2 - w*h`, so a platelet weighs about twice a full-frame object. Shown by giving the
    same relative error to a small box and a large one."""
    loss = Yolo3Loss(anchors=ANCHORS, n_class=1)
    scores = []
    for box in ([[0, 0, 400, 400]], [[0, 0, 24, 24]]):
        targets = targets_for(box, [0], n_class=1)
        wrong = [p.detach() + 0.5 for p in perfect(targets)]
        scores.append(loss(wrong, targets).coordinates.item())
    large, small = scores
    assert small > large


# --- the ignore mask --------------------------------------------------------------------


def test_a_cell_predicting_the_real_object_is_left_alone():
    """The detail most reimplementations drop. A cell beside the assigned one often
    predicts a box that overlaps the truth - it has seen the same pixels - and training it
    towards zero objectness teaches the model to suppress correct answers."""
    one_anchor = (((0.25, 0.25),), ((0.10, 0.10),), ((0.04, 0.04),))
    targets = targets_for([[104, 104, 208, 208]], [0], n_class=1, anchors=one_anchor)

    # Every cell predicts a box of its own anchor's size, centred on itself.
    predictions = []
    for target in targets:
        prediction = torch.zeros_like(target)
        prediction[..., 4] = 5.0
        predictions.append(prediction)

    loss = Yolo3Loss(anchors=one_anchor, n_class=1, ignore_threshold=0.5)
    mask = loss._ignore_mask(0, predictions[0], loss._truth_boxes(targets))
    assert mask.sum() > 0, "some cell should be predicting the real object"

    # And the finest grid, whose anchors are far smaller than the object, has none.
    fine = loss._ignore_mask(2, predictions[2], loss._truth_boxes(targets))
    assert fine.sum() == 0


def test_the_mask_changes_which_cells_are_supervised():
    """Asserted on the mechanism rather than on the scalar.

    An earlier version of this compared the no-object loss with the mask on and off and
    expected a difference. There is none to speak of, and that is a consequence of
    normalising that term by the cells it supervises rather than by the object count:
    dropping three cells out of three and a half thousand barely moves a mean, where it
    moved a sum visibly. The mask still does its work - an excluded cell contributes no
    gradient at all - and the count is where that shows.
    """
    one_anchor = (((0.25, 0.25),), ((0.10, 0.10),), ((0.04, 0.04),))
    targets = targets_for([[104, 104, 208, 208]], [0], n_class=1, anchors=one_anchor)
    predictions = []
    for target in targets:
        prediction = torch.zeros_like(target)
        prediction[..., 4] = 5.0
        predictions.append(prediction)

    truth = Yolo3Loss(anchors=one_anchor, n_class=1)._truth_boxes(targets)
    on = Yolo3Loss(anchors=one_anchor, n_class=1, ignore_threshold=0.5)
    off = Yolo3Loss(anchors=one_anchor, n_class=1, ignore_threshold=1.0)

    supervised_on = (~on._ignore_mask(0, predictions[0], truth)).sum().item()
    supervised_off = (~off._ignore_mask(0, predictions[0], truth)).sum().item()
    assert supervised_off > supervised_on


def test_the_mask_is_not_differentiated_through():
    """It decides *whether* a position is supervised, and a decision is not a quantity to
    take a gradient of."""
    targets = targets_for([[40, 40, 140, 140]], [0])
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3)
    prediction = perfect(targets)[0].detach().requires_grad_(True)
    mask = loss._ignore_mask(0, prediction, loss._truth_boxes(targets))
    assert mask.dtype == torch.bool
    assert prediction.grad is None


def test_chunking_the_overlaps_does_not_change_the_mask(monkeypatch):
    """The chunk exists to bound memory, so it has to be invisible in the answer.

    Asserted against the unchunked result rather than against a stored mask, because the
    claim is an equivalence and not a particular shape. Several truths and a chunk smaller
    than one grid, so every boundary case is crossed.
    """
    import pyplatypus.detection.loss as module

    boxes = [
        [40, 40, 140, 140],
        [200, 210, 260, 280],
        [300, 20, 390, 100],
        [10, 300, 120, 400],
        [180, 180, 200, 205],
    ]
    targets = targets_for(boxes, [0, 1, 2, 0, 1])
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3, ignore_threshold=0.3)
    truth = loss._truth_boxes(targets)
    predictions = [p * 0.3 for p in perfect(targets)]

    monkeypatch.setattr(module, "_IOU_CHUNK", 10**9)
    whole = [loss._ignore_mask(i, predictions[i], truth) for i in range(3)]
    monkeypatch.setattr(module, "_IOU_CHUNK", 7)
    chunked = [loss._ignore_mask(i, predictions[i], truth) for i in range(3)]

    assert any(m.any() for m in whole), "nothing was ignored, so there is nothing to compare"
    for one, other in zip(whole, chunked):
        assert torch.equal(one, other)


# --- truth recovered from the targets ---------------------------------------------------


def test_the_truth_boxes_are_read_back_out_of_the_targets():
    """So the loss needs nothing the model's own target does not already carry - which
    works because `encode` and `decode` are exact inverses."""
    boxes = [[40, 40, 140, 140], [250, 100, 290, 150]]
    targets = targets_for(boxes, [0, 2])
    recovered = Yolo3Loss(anchors=ANCHORS, n_class=3)._truth_boxes(targets)[0]

    assert len(recovered) == 2
    # Normalised corners, so scale back up to compare.
    got = sorted((recovered * 416).tolist())
    want = sorted(boxes)
    for one, other in zip(got, want):
        assert all(abs(a - b) < 0.5 for a, b in zip(one, other)), (one, other)


def test_an_image_with_nothing_in_it_contributes_only_background():
    targets = targets_for([], [], n_class=3)
    parts = Yolo3Loss(anchors=ANCHORS, n_class=3)(
        [torch.zeros_like(t) for t in targets], targets
    ).as_dict()
    assert parts["coordinates"] == 0.0
    assert parts["objectness"] == 0.0
    assert parts["classes"] == 0.0
    assert parts["no_object"] > 0.0


# --- refusals ---------------------------------------------------------------------------


def test_a_grid_count_that_does_not_match_the_anchors_is_refused():
    targets = targets_for([[40, 40, 140, 140]], [0])
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3)
    with pytest.raises(DetectionError, match="grids predicted"):
        loss(targets[:2], targets)


def test_a_shape_mismatch_between_prediction_and_target_is_refused():
    targets = targets_for([[40, 40, 140, 140]], [0])
    wrong = [torch.zeros_like(t) for t in targets]
    wrong[1] = torch.zeros((1, 26, 26, 2, 9))
    with pytest.raises(DetectionError, match="grid 1"):
        Yolo3Loss(anchors=ANCHORS, n_class=3)(wrong, targets)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_an_impossible_ignore_threshold_is_refused(threshold):
    with pytest.raises(DetectionError, match="ignore_threshold"):
        Yolo3Loss(anchors=ANCHORS, n_class=3, ignore_threshold=threshold)


def test_unequal_anchor_counts_are_refused():
    with pytest.raises(DetectionError, match="same number of anchors"):
        Yolo3Loss(anchors=(((0.2, 0.2), (0.3, 0.3)), ((0.1, 0.1),), ((0.05, 0.05),)), n_class=1)


# --- the join with the model -------------------------------------------------------------


def test_the_model_and_the_loss_fit_together_and_training_reduces_it():
    """The join nobody tests until a training run fails.

    Clipped, and at the rate the example script uses, over enough steps to see the trend.

    Two things were learned writing this. One unclipped step at 1e-3 took the loss from
    9.1 to 219 - not a fault in the objective, but worth knowing: from a random start the
    no-object term pushes ten thousand cells at once and the gradient is large, so a
    detector trained without `clip_grad_norm_` diverges immediately. And even clipped, the
    **first few steps make it worse** - 9.1, 19.1, 54.6 - before it falls to under 4 by
    step fourteen, because Adam has no accumulated second moment yet and takes near-maximal
    steps. Five steps was the first version of this test and it failed for that reason. A
    real run hides it: at the epoch level the climb is inside the first batches.
    """
    from pyplatypus.detection.yolo3 import build_yolo3

    torch.manual_seed(0)
    model = build_yolo3(n_class=3, anchors_per_grid=2)
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3, input_shape=(128, 128))
    targets = targets_for([[20, 20, 60, 60]], [1], size=128)
    image = torch.randn(1, 3, 128, 128)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = loss(model(image), targets).total.item()
    for _ in range(15):
        parts = loss(model(image), targets)
        optimiser.zero_grad()
        parts.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
    after = loss(model(image), targets).total.item()

    assert after < before, (before, after)


def test_an_unclipped_step_can_make_it_worse():
    """Stated as a test so the clipping in the example script is not mistaken for
    decoration. This is the behaviour that makes it necessary."""
    from pyplatypus.detection.yolo3 import build_yolo3

    torch.manual_seed(0)
    model = build_yolo3(n_class=3, anchors_per_grid=2)
    loss = Yolo3Loss(anchors=ANCHORS, n_class=3, input_shape=(128, 128))
    targets = targets_for([[20, 20, 60, 60]], [1], size=128)
    image = torch.randn(1, 3, 128, 128)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = loss(model(image), targets)
    optimiser.zero_grad()
    before.total.backward()
    optimiser.step()
    after = loss(model(image), targets).total.item()
    assert after > before.total.item()
