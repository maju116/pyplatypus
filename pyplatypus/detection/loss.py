"""The YOLOv3 objective, in four parts, and the mask that stops it punishing success.

A detection loss is not one quantity. Each cell of each grid predicts a box, an objectness
and a class distribution, and almost every cell contains nothing - so the three parts have
to be weighted against each other and the overwhelming majority of positions contribute
only "there is nothing here".

**The ignore mask is the part worth understanding.** A cell that was not assigned a truth
box is supposed to predict "no object". But a cell beside the assigned one often predicts
a box that overlaps the real object well - it has seen the same pixels. Training it towards
zero objectness teaches the model to suppress correct answers, and the symptom is a
detector whose confidences collapse as training continues. YOLOv3's answer: a non-assigned
cell whose predicted box overlaps some truth by more than `ignore_threshold` contributes
**nothing** to the objectness loss. Not a zero target, not a one - it is excluded. The old
package had this and it is the detail most reimplementations drop.

**The four terms are normalised to be comparable.** Coordinates, objectness and classes
are summed over the assigned cells and divided by how many there were - a mean per object.
The no-object term is a mean over the cells it supervises, which is not the same thing and
was wrong in the first version of this file: it sums over every empty cell, so dividing by
the object count gave an image with one object a no-object loss of 4915 against a
coordinate loss of 5.5. At that ratio the only thing a model can learn is to answer
"nothing here" everywhere, which is the classic way a detection run fails while the loss
falls convincingly.

**Small boxes are weighted up**, by `2 - w * h` in normalised units. Without it the
coordinate loss is dominated by large objects, since an equal fractional error on a big box
is a larger absolute one; with it a platelet and a white cell matter comparably. The factor
is between 1 and 2 by construction, so it rescales rather than reorders.

Targets come from `encode`, which stores the within-cell offset in `[0, 1)` and the
log-ratio to the anchor - not the pre-activation values the old package stored, which went
to `-inf` whenever a centre landed on a cell boundary. So this applies the sigmoid itself
and compares against a finite number, which is the whole reason that change was made.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pyplatypus.detection.encode import COCO_ANCHORS, STRIDES, grid_shapes
from pyplatypus.detection.metrics import DetectionError


@dataclass(frozen=True)
class LossParts:
    """The objective broken out, because one number cannot say what is wrong.

    A run whose coordinate loss falls while its objectness loss does not is finding the
    right places and refusing to commit; the reverse is confident nonsense. Summed they
    are the number to minimise, separately they are the number to read.
    """

    total: torch.Tensor
    coordinates: torch.Tensor
    objectness: torch.Tensor
    no_object: torch.Tensor
    classes: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total),
            "coordinates": float(self.coordinates),
            "objectness": float(self.objectness),
            "no_object": float(self.no_object),
            "classes": float(self.classes),
        }


class Yolo3Loss(nn.Module):
    """Logits in from the model, targets in from `encode`, one scalar out.

    `anchors` must be the same ones the targets were encoded with. Nothing can check that
    - the arrays carry no record of it - so it is the one thing a caller has to keep
    straight, and getting it wrong trains a model whose boxes are a fixed factor too large
    or too small with no other symptom.
    """

    def __init__(self, *, anchors=COCO_ANCHORS, n_class: int = 80,
                 input_shape: tuple[int, int] = (416, 416),
                 ignore_threshold: float = 0.5, coordinate_weight: float = 1.0,
                 objectness_weight: float = 1.0, no_object_weight: float = 1.0,
                 class_weight: float = 1.0, strides: tuple[int, ...] = STRIDES):
        super().__init__()
        if not 0 < ignore_threshold <= 1:
            raise DetectionError(
                f"ignore_threshold must be above 0 and at most 1; got {ignore_threshold}"
            )
        self.anchors = tuple(tuple(tuple(float(v) for v in pair) for pair in group)
                             for group in anchors)
        self.per_grid = len(self.anchors[0])
        if any(len(group) != self.per_grid for group in self.anchors):
            raise DetectionError(
                "every grid needs the same number of anchors; the head's width is "
                "anchors_per_grid * (n_class + 5)"
            )
        self.n_class = n_class
        self.input_shape = (int(input_shape[0]), int(input_shape[1]))
        self.ignore_threshold = float(ignore_threshold)
        self.weights = {
            "coordinates": float(coordinate_weight),
            "objectness": float(objectness_weight),
            "no_object": float(no_object_weight),
            "classes": float(class_weight),
        }
        self.strides = strides
        self.shapes = grid_shapes(self.input_shape, scales=len(self.anchors),
                                  strides=strides)

    def forward(self, predictions, targets) -> LossParts:
        if len(predictions) != len(self.anchors):
            raise DetectionError(
                f"{len(predictions)} grids predicted but the anchors describe "
                f"{len(self.anchors)}"
            )
        if len(targets) != len(self.anchors):
            raise DetectionError(
                f"{len(targets)} grids of target but the anchors describe "
                f"{len(self.anchors)}"
            )

        device = predictions[0].device
        zero = torch.zeros((), device=device)
        coordinates, objectness, no_object, classes = zero, zero, zero, zero

        # Every truth box in the batch, per image, recovered from the targets themselves -
        # which works because `encode` and `decode` are exact inverses, and means the loss
        # needs nothing the model's own target does not already carry.
        truth = self._truth_boxes(targets)

        for grid, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
            if prediction.shape != target.shape:
                raise DetectionError(
                    f"grid {grid}: the model predicted {tuple(prediction.shape)} and the "
                    f"target is {tuple(target.shape)}"
                )
            parts = self._one_grid(grid, prediction, target, truth)
            coordinates = coordinates + parts[0]
            objectness = objectness + parts[1]
            no_object = no_object + parts[2]
            classes = classes + parts[3]

        total = (self.weights["coordinates"] * coordinates
                 + self.weights["objectness"] * objectness
                 + self.weights["no_object"] * no_object
                 + self.weights["classes"] * classes)
        return LossParts(total=total, coordinates=coordinates, objectness=objectness,
                         no_object=no_object, classes=classes)

    # ------------------------------------------------------------------ internals
    def _one_grid(self, grid: int, prediction: torch.Tensor, target: torch.Tensor,
                  truth: list[torch.Tensor]):
        device = prediction.device
        anchors = torch.tensor(self.anchors[grid], dtype=prediction.dtype, device=device)

        has_object = target[..., 4] > 0.5
        count = has_object.sum().clamp(min=1)

        # Coordinates, only where there is something to locate.
        offsets = prediction[..., 0:2]
        sizes = prediction[..., 2:4]
        target_offsets = target[..., 0:2]
        target_sizes = target[..., 2:4]

        # 2 - w*h in normalised units, so a platelet weighs about twice a full-frame object.
        box_w = torch.exp(target_sizes[..., 0]) * anchors[:, 0]
        box_h = torch.exp(target_sizes[..., 1]) * anchors[:, 1]
        scale = (2.0 - (box_w * box_h).clamp(0, 1))[has_object]

        if has_object.any():
            offset_loss = F.binary_cross_entropy_with_logits(
                offsets[has_object], target_offsets[has_object], reduction="none"
            ).sum(dim=-1)
            size_loss = F.mse_loss(
                sizes[has_object], target_sizes[has_object], reduction="none"
            ).sum(dim=-1)
            coordinates = ((offset_loss + size_loss) * scale).sum() / count
            classes = F.binary_cross_entropy_with_logits(
                prediction[has_object][..., 5:], target[has_object][..., 5:],
                reduction="sum"
            ) / count
            objectness = F.binary_cross_entropy_with_logits(
                prediction[..., 4][has_object], torch.ones_like(prediction[..., 4][has_object]),
                reduction="sum"
            ) / count
        else:
            coordinates = torch.zeros((), device=device)
            classes = torch.zeros((), device=device)
            objectness = torch.zeros((), device=device)

        ignore = self._ignore_mask(grid, prediction, truth)
        punish = (~has_object) & (~ignore)
        if punish.any():
            # Averaged over the cells it supervises, not over the objects. Dividing this
            # by the object count was the first version and it is the classic way to make
            # a detector that only ever says "nothing here": the term sums over thousands
            # of empty cells, so one object in an image gave a no-object loss of 4915
            # against a coordinate loss of 5.5, and nothing else could be heard. As a mean
            # the four terms are comparable and `no_object_weight` means something.
            no_object = F.binary_cross_entropy_with_logits(
                prediction[..., 4][punish], torch.zeros_like(prediction[..., 4][punish]),
                reduction="mean"
            )
        else:
            no_object = torch.zeros((), device=device)

        return coordinates, objectness, no_object, classes

    def _ignore_mask(self, grid: int, prediction: torch.Tensor,
                     truth: list[torch.Tensor]) -> torch.Tensor:
        """Which empty cells are predicting something real and must be left alone.

        Computed without gradient on purpose: this decides *whether* a position is
        supervised, and a decision is not a quantity to differentiate through.
        """
        with torch.no_grad():
            boxes = self._decode_boxes(grid, prediction)       # (n, h, w, a, 4)
            batch = boxes.shape[0]
            mask = torch.zeros(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
            for index in range(batch):
                if truth[index].numel() == 0:
                    continue
                flat = boxes[index].reshape(-1, 4)
                truths = truth[index].to(boxes.device)
                best = torch.zeros(len(flat), device=boxes.device)
                # In chunks, because the pairwise matrix is cells x truths and both grow:
                # the finest grid of a 608 input has 17,328 cells and a blood smear has up
                # to 45 objects, which is where this ran out of memory at batch 8. The
                # chunk bounds it without changing the answer.
                for start in range(0, len(flat), _IOU_CHUNK):
                    piece = flat[start:start + _IOU_CHUNK]
                    best[start:start + _IOU_CHUNK] = _iou(piece, truths).max(dim=1).values
                mask[index] = (best > self.ignore_threshold).reshape(boxes.shape[1:-1])
            return mask

    def _decode_boxes(self, grid: int, prediction: torch.Tensor) -> torch.Tensor:
        """Predicted boxes in normalised corner form, for the mask only."""
        device = prediction.device
        grid_h, grid_w = self.shapes[grid]
        anchors = torch.tensor(self.anchors[grid], dtype=prediction.dtype, device=device)

        rows = torch.arange(grid_h, device=device).view(1, grid_h, 1, 1)
        columns = torch.arange(grid_w, device=device).view(1, 1, grid_w, 1)
        centre_x = (torch.sigmoid(prediction[..., 0]) + columns) / grid_w
        centre_y = (torch.sigmoid(prediction[..., 1]) + rows) / grid_h
        width = torch.exp(prediction[..., 2].clamp(max=10)) * anchors[:, 0]
        height = torch.exp(prediction[..., 3].clamp(max=10)) * anchors[:, 1]
        return torch.stack([centre_x - width / 2, centre_y - height / 2,
                            centre_x + width / 2, centre_y + height / 2], dim=-1)

    def _truth_boxes(self, targets) -> list[torch.Tensor]:
        """The batch's true boxes in normalised corner form, read back out of the targets."""
        batch = targets[0].shape[0]
        out = [[] for _ in range(batch)]
        for grid, target in enumerate(targets):
            grid_h, grid_w = self.shapes[grid]
            anchors = torch.tensor(self.anchors[grid], dtype=target.dtype,
                                   device=target.device)
            index = (target[..., 4] > 0.5).nonzero(as_tuple=False)
            for image, row, column, slot in index.tolist():
                values = target[image, row, column, slot]
                centre_x = (values[0] + column) / grid_w
                centre_y = (values[1] + row) / grid_h
                width = torch.exp(values[2]) * anchors[slot, 0]
                height = torch.exp(values[3]) * anchors[slot, 1]
                out[image].append(torch.stack([centre_x - width / 2, centre_y - height / 2,
                                               centre_x + width / 2, centre_y + height / 2]))
        return [torch.stack(rows) if rows else torch.zeros((0, 4), device=targets[0].device)
                for rows in out]


#: How many predicted boxes to compare against the truths at once. The pairwise matrix is
#: chunk x truths, so this bounds the ignore mask's memory independently of the input size.
_IOU_CHUNK = 4096


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of corner boxes."""
    left = torch.maximum(a[:, None, 0], b[None, :, 0])
    top = torch.maximum(a[:, None, 1], b[None, :, 1])
    right = torch.minimum(a[:, None, 2], b[None, :, 2])
    bottom = torch.minimum(a[:, None, 3], b[None, :, 3])
    overlap = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - overlap
    return torch.where(union > 0, overlap / union, torch.zeros_like(overlap))
