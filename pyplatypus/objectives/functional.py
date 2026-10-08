"""The maths, as pure functions on tensors.

PLAN.md rule 3, made concrete in one line: `spatial_dims` returns every axis except
batch and channel, and every overlap statistic reduces over exactly those. Written this
way, Dice on a 256x256 image and Dice on a 128^3 volume are the same function.

Convention everywhere below:
    logits  (B, C, *spatial)  raw model output, never activated
    target  (B, C, *spatial)  one-hot, or (B, *spatial) class indices
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7


def spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    """Axis 0 is batch, axis 1 is channel, everything after that is space."""
    return tuple(range(2, x.ndim))


def as_onehot(target: torch.Tensor, n_class: int) -> torch.Tensor:
    """Accept either one-hot or class indices, always return one-hot."""
    if target.ndim > 1 and target.shape[1] == n_class and target.is_floating_point():
        return target
    onehot = F.one_hot(target.long(), n_class)
    # one_hot appends the class axis; move it to position 1.
    return onehot.permute(0, -1, *range(1, target.ndim)).float()


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=1)


def _dilate(x: torch.Tensor) -> torch.Tensor:
    """Grey dilation with a full 3-wide structuring element, at any rank."""
    pool = F.max_pool2d if x.ndim == 4 else F.max_pool3d
    return pool(x, kernel_size=3, stride=1, padding=1)


def _erode(x: torch.Tensor) -> torch.Tensor:
    """Grey erosion, which is dilation of the complement.

    `max_pool` pads with -inf, so the complement is padded with +inf and the border is
    never eroded from outside - the same convention as `scipy.ndimage.binary_erosion`
    with `border_value=1`, which is what the test compares against.
    """
    return -_dilate(-x)


def soft_skeleton(x: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """The soft skeleton of Shit et al., CVPR 2021: iterated min and max pooling.

    Morphological skeletonisation by openings, written with pooling so that every step is
    differentiable. On a binary input it is the ordinary morphological skeleton; on
    probabilities it degrades smoothly, which is what lets the same function serve a metric
    now and a loss later. A hard skeleton would have been cheaper and is not comparable
    with the paper.

    `iterations` must reach the half-width of the thickest structure, or its core is never
    peeled down to a centreline and the skeleton keeps a solid middle. Five covers a
    ten-pixel object, which is well past a retinal vessel at one to five.

    Args:
        x: `(B, C, *spatial)`, 0-1. Rank-generic: 2D and 3D take the same path.
        iterations: how many times to peel.

    Returns:
        The skeleton, same shape, 0-1.

    >>> import torch
    >>> line = torch.zeros(1, 1, 7, 7)
    >>> line[0, 0, 3, 1:6] = 1.0
    >>> skeleton = soft_skeleton(line)
    >>> bool(torch.equal(skeleton, line))     # one pixel wide already
    True
    """
    opened = _dilate(_erode(x))
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = _erode(x)
        opened = _dilate(_erode(x))
        peeled = F.relu(x - opened)
        # Soft union: add what this round found and the running skeleton did not have.
        skeleton = skeleton + F.relu(peeled - skeleton * peeled)
    return skeleton


def cldice_from_masks(
    prediction: torch.Tensor, target: torch.Tensor, iterations: int = 5, smooth: float = 1.0
):
    """Centerline Dice, per sample and class.

    Dice asks whether the pixels coincide. This asks whether the *structure* does: each
    mask is scored against the other's skeleton, so a vessel one pixel out of place scores
    well and a vessel broken in half does not - which is the distinction Dice cannot make
    and the reason a 1-5 pixel structure needs its own metric.

    There is no `*_from_overlaps` form and there cannot be: a skeleton is a property of a
    whole mask, not of its overlap counts, so this cannot be accumulated from the pieces of
    a tiled image the way Dice is.
    """
    dims = spatial_dims(prediction)
    skeleton_p = soft_skeleton(prediction, iterations)
    skeleton_t = soft_skeleton(target, iterations)
    # A structure thicker than `iterations` can peel is never reduced to a centreline and
    # its skeleton comes back **empty** - at which point the ratios above are smooth/smooth
    # and the class scores a perfect 1.0. Measured: a 24-pixel square at iterations=1 gives
    # clDice 1.0 to a prediction whose Dice is 0.0017. NaN instead, which the metric refuses
    # on rather than averaging in.
    degenerate = ((skeleton_p.sum(dims) == 0) & (prediction.sum(dims) > 0)) | (
        (skeleton_t.sum(dims) == 0) & (target.sum(dims) > 0)
    )
    # Topology precision: how much of the prediction's centreline lies inside the truth.
    precision = (skeleton_p * target).sum(dims).add(smooth) / skeleton_p.sum(dims).add(smooth)
    # Topology sensitivity: how much of the truth's centreline the prediction covers.
    sensitivity = (skeleton_t * prediction).sum(dims).add(smooth) / skeleton_t.sum(dims).add(smooth)
    score = 2 * precision * sensitivity / (precision + sensitivity + EPS)
    return torch.where(degenerate, torch.tensor(float("nan"), device=score.device), score)


def overlaps(
    probs: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """True positives, false positives and false negatives, per sample and class."""
    dims = spatial_dims(probs)
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    return tp, fp, fn


# The three coefficients below are written twice over: once as a formula on overlap
# statistics, and once as the convenient call on tensors. Anything that can sum TP, FP and
# FN itself - scoring a case whose tiles arrived separately, for instance - needs the
# formula without the tensors, and a metric must not have two implementations that can
# drift apart. So the formula lives in `*_from_overlaps` and everything else calls it.
def dice_from_overlaps(tp, fp, fn, smooth: float = 1.0):
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)


def iou_from_overlaps(tp, fp, fn, smooth: float = 1.0):
    return (tp + smooth) / (tp + fp + fn + smooth)


def tversky_from_overlaps(tp, fp, fn, alpha: float = 0.5, smooth: float = 1.0):
    beta = 1.0 - alpha
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def dice_coefficient(
    probs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    return dice_from_overlaps(*overlaps(probs, target), smooth)


def iou_coefficient(probs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    return iou_from_overlaps(*overlaps(probs, target), smooth)


def tversky_coefficient(
    probs: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, smooth: float = 1.0
) -> torch.Tensor:
    """TP / (TP + alpha*FN + beta*FP), with beta = 1 - alpha.

    alpha weights false negatives, so raising it punishes missed foreground and pushes
    the model towards recall.

    At alpha = 0.5 this is Dice - but only exactly so without smoothing. Multiplying
    through by two gives (2TP + 2s) / (2TP + FN + FP + 2s), so with smoothing it equals
    `dice_coefficient` at twice the smoothing. Easy to trip over when comparing runs.
    """
    return tversky_from_overlaps(*overlaps(probs, target), alpha, smooth)


def cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Categorical cross-entropy against a one-hot target, averaged over everything."""
    log_probs = logits.log_softmax(dim=1)
    if label_smoothing > 0:
        n_class = logits.shape[1]
        target = target * (1 - label_smoothing) + label_smoothing / n_class
    return -(target * log_probs).sum(dim=1).mean()


def focal(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """Down-weights the pixels the model already gets right, so the hard ones dominate."""
    log_probs = logits.log_softmax(dim=1)
    probs = log_probs.exp()
    weight = (1 - probs).clamp_min(0).pow(gamma)
    if alpha is not None:
        # alpha on the foreground classes, 1 - alpha on the background class.
        weights = torch.full((logits.shape[1],), alpha, device=logits.device)
        weights[0] = 1 - alpha
        weight = weight * weights.view(1, -1, *([1] * (logits.ndim - 2)))
    return -(target * weight * log_probs).sum(dim=1).mean()


def combo(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    ce_ratio: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Taghanaki's Combo loss: a lopsided cross-entropy blended with Dice.

    `ce_ratio` tilts the cross-entropy between punishing false negatives (towards 1) and
    false positives (towards 0); `alpha` sets how much of the total each half provides.
    """
    probs = probabilities(logits).clamp(EPS, 1 - EPS)
    weighted_ce = -(
        ce_ratio * target * probs.log() + (1 - ce_ratio) * (1 - target) * (1 - probs).log()
    ).mean()
    dice_loss = 1 - dice_coefficient(probabilities(logits), target, smooth).mean()
    return alpha * weighted_ce + (1 - alpha) * dice_loss


def _lovasz_grad(sorted_target: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard index (Berman et al., 2018)."""
    total = sorted_target.sum()
    intersection = total - sorted_target.cumsum(0)
    union = total + (1 - sorted_target).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(EPS)
    if len(sorted_target) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(
    logits: torch.Tensor, target: torch.Tensor, per_image: bool = False
) -> torch.Tensor:
    """A convex surrogate for IoU that is actually differentiable.

    Classes absent from a batch are skipped rather than scored as perfect, which is what
    'present' means in the reference implementation.
    """
    probs = probabilities(logits)

    def one(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n_class = p.shape[0]
        flat_p = p.reshape(n_class, -1).transpose(0, 1)
        flat_t = t.reshape(n_class, -1).transpose(0, 1)
        losses = []
        for c in range(n_class):
            foreground = flat_t[:, c]
            if foreground.sum() == 0:
                continue
            errors = (foreground - flat_p[:, c]).abs()
            errors_sorted, order = torch.sort(errors, dim=0, descending=True)
            losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground[order])))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    if per_image:
        return torch.stack([one(p, t) for p, t in zip(probs, target, strict=True)]).mean()
    # Treating the batch as one image: concatenate along space, keeping the class axis.
    merged_p = probs.transpose(0, 1).reshape(probs.shape[1], -1)
    merged_t = target.transpose(0, 1).reshape(target.shape[1], -1)
    return one(merged_p, merged_t)
