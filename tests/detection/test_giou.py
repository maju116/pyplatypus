"""`box_loss="giou"` — the generalized-IoU box term.

The whole point of GIoU over IoU is that it has a gradient where the boxes do not touch,
which is where a detector is most wrong; and the whole point of it over YOLOv3's own
offsets term is that it can reach zero, so a converged loss is readable. Both are tested
here rather than described.
"""

import numpy as np
import pytest
import torch

from pyplatypus.detection.encode import COCO_ANCHORS, encode
from pyplatypus.detection.loss import Yolo3Loss, _giou, _iou
from pyplatypus.detection.metrics import DetectionError

# Boxes and the values torchvision's `generalized_box_iou` returns for them, baked in so
# this suite needs no torchvision - the same arrangement the detection metrics use with
# pycocotools. Regenerate by taking the diagonal of `generalized_box_iou(A, B)`.
A = torch.tensor([
    [0.534923, 0.198803, 0.987151, 0.262634],
    [0.659212, 0.656890, 1.261163, 0.741562],
    [0.232762, 0.425061, 0.782488, 0.692965],
    [0.207086, 0.629736, 0.681403, 1.066645],
    [0.365316, 0.851268, 0.697874, 1.227207],
    [0.854943, 0.550935, 1.402766, 0.948089],
    [0.286839, 0.206319, 0.766003, 0.294252],
    [0.445090, 0.359286, 0.978656, 0.782537],
])
B = torch.tensor([
    [0.626651, 0.569073, 1.088768, 0.608939],
    [0.743733, 0.959217, 0.818559, 1.518859],
    [0.388742, 0.221446, 1.004914, 0.523619],
    [0.374202, 0.195258, 0.457156, 0.523455],
    [0.740524, 0.252880, 0.920958, 0.572303],
    [0.233150, 0.931414, 0.699990, 1.384216],
    [0.957538, 0.557507, 1.242359, 0.910536],
    [0.413419, 0.435458, 0.815060, 0.520311],
])
TORCHVISION = torch.tensor([
    -0.791817009, -0.821061671, -0.058988228, -0.432734549,
    -0.662607968, -0.559873164, -0.787954509, 0.092576548,
])


def test_giou_agrees_with_torchvision():
    """Verified against `torchvision.ops.generalized_box_iou` rather than against a
    rearrangement of the same formula, which would only prove the algebra was copied.

    Computed side by side in float32 the two agree exactly, to 0.0e+00. The tolerance here
    is for the baked literals: writing a float32 out as a decimal string and reading it
    back is not the identity, and the worst case over these eight was 1.01e-06 - which is
    the round trip, not the formula.
    """
    torch.testing.assert_close(_giou(A, B), TORCHVISION, rtol=0, atol=1e-5)


def test_identical_boxes_score_one_and_separation_drives_it_negative():
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    assert _giou(box, box).item() == pytest.approx(1.0)

    near = torch.tensor([[0.55, 0.1, 0.95, 0.5]])
    far = torch.tensor([[0.90, 0.90, 1.30, 1.30]])
    # Both are disjoint, so IoU cannot tell them apart; GIoU must.
    assert _iou(box, near).item() == 0.0
    assert _iou(box, far).item() == 0.0
    assert _giou(box, near).item() > _giou(box, far).item()
    assert _giou(box, far).item() < 0


def test_giou_has_a_gradient_where_iou_has_none():
    """The reason this exists. Two boxes that do not touch give IoU exactly zero for every
    position, so its gradient is zero and the optimiser is told nothing at all."""
    truth = torch.tensor([[0.1, 0.1, 0.3, 0.3]])

    predicted = torch.tensor([[0.7, 0.7, 0.9, 0.9]], requires_grad=True)
    (1.0 - _iou(predicted, truth)).sum().backward()
    iou_gradient = predicted.grad.abs().sum().item()

    predicted = torch.tensor([[0.7, 0.7, 0.9, 0.9]], requires_grad=True)
    (1.0 - _giou(predicted, truth)).sum().backward()
    giou_gradient = predicted.grad.abs().sum().item()

    assert iou_gradient == 0.0
    assert giou_gradient > 0.1


# --- the loss ---------------------------------------------------------------------------

def _targets_and_a_perfect_prediction():
    """Targets for three boxes, and logits that decode to exactly those boxes."""
    boxes = np.array([[40.0, 50.0, 150.0, 170.0],
                      [220.0, 60.0, 300.0, 140.0],
                      [10.0, 300.0, 70.0, 360.0]])
    enc = encode(boxes, [0, 1, 2], anchors=COCO_ANCHORS, input_shape=(416, 416), n_class=3)
    assert enc.placed == 3, "the fixture is only meaningful if every box was placed"
    targets = [torch.tensor(t, dtype=torch.float32)[None] for t in enc.targets]

    predictions = []
    for target in targets:
        p = torch.zeros_like(target)
        here = target[..., 4] > 0.5
        eps = 1e-6
        p[..., 0] = torch.logit(target[..., 0].clamp(eps, 1 - eps))   # undo the sigmoid
        p[..., 1] = torch.logit(target[..., 1].clamp(eps, 1 - eps))
        p[..., 2] = target[..., 2]                                     # log sizes as written
        p[..., 3] = target[..., 3]
        p[..., 4] = torch.where(here, 20.0, -20.0)
        p[..., 5:] = torch.where(target[..., 5:] > 0.5, 20.0, -20.0)
        predictions.append(p)
    return predictions, targets


def _loss(mode):
    return Yolo3Loss(anchors=COCO_ANCHORS, n_class=3, input_shape=(416, 416), box_loss=mode)


def test_a_perfect_prediction_costs_nothing_under_giou_and_something_under_offsets():
    """The property that decides whether a converged loss can be read.

    Cross-entropy against a soft target bottoms out at that target's entropy, so the
    offsets term cannot reach zero however right the boxes are - and a run sitting at its
    floor is then indistinguishable from one that has stalled.
    """
    predictions, targets = _targets_and_a_perfect_prediction()

    giou = _loss("giou")(predictions, targets).as_dict()
    offsets = _loss("offsets")(predictions, targets).as_dict()

    assert giou["coordinates"] == pytest.approx(0.0, abs=1e-5)
    assert giou["loss"] == pytest.approx(0.0, abs=1e-5)
    assert offsets["coordinates"] > 4.0          # measured: 4.2138 on this fixture
    assert offsets["loss"] == pytest.approx(offsets["coordinates"], abs=1e-5)


def test_a_wrong_box_costs_more_than_a_right_one():
    """Guards against the term being zero for reasons other than being correct - a
    constant would pass the test above."""
    predictions, targets = _targets_and_a_perfect_prediction()
    nudged = [p.clone() for p in predictions]
    for p in nudged:
        p[..., 2] += 0.5                          # every box half a log-unit too wide

    right = _loss("giou")(predictions, targets).as_dict()["coordinates"]
    wrong = _loss("giou")(nudged, targets).as_dict()["coordinates"]
    assert wrong > right + 0.05


def test_the_giou_term_carries_a_gradient_to_the_coordinate_logits():
    predictions, targets = _targets_and_a_perfect_prediction()
    nudged = []
    for p in predictions:
        q = p.clone()
        q[..., 0:4] += 0.3
        nudged.append(q.requires_grad_(True))

    _loss("giou")(nudged, targets).total.backward()
    for q in nudged:
        assert q.grad is not None
    assert sum(q.grad[..., 0:4].abs().sum().item() for q in nudged) > 0


def test_offsets_is_the_default_so_published_weights_keep_their_objective():
    assert Yolo3Loss(n_class=3).box_loss == "offsets"
    predictions, targets = _targets_and_a_perfect_prediction()
    default = Yolo3Loss(anchors=COCO_ANCHORS, n_class=3, input_shape=(416, 416))
    assert (default(predictions, targets).as_dict()
            == _loss("offsets")(predictions, targets).as_dict())


def test_an_unknown_box_loss_is_refused_by_name():
    with pytest.raises(DetectionError, match="box_loss is 'offsets' or 'giou'"):
        Yolo3Loss(n_class=3, box_loss="ciou")
