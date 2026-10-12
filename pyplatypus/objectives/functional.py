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


def probabilities(logits: torch.Tensor, *, multilabel: bool = False) -> torch.Tensor:
    """Logits to probabilities, and the one line where mutual exclusivity lives.

    A softmax across channels says the channels are alternatives and makes them sum to one;
    a sigmoid per channel says each is its own yes-or-no. Everything overlap-based above
    reads this function and nothing else, so nested regions reach Dice, IoU, Tversky and the
    surface term through this argument alone.
    """
    return logits.sigmoid() if multilabel else logits.softmax(dim=1)


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


def binary_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy for channels that are independent questions, not alternatives.

    **The reduction differs from the categorical form, and deliberately.** There, the target
    is one-hot, so `sum(dim=1)` picks out the one true class and the result is the negative
    log probability of that class. Here every channel contributes a term, so a sum would make
    the loss grow with the number of regions - three regions would read three times the scale
    of one, and `cce_weight` in a blended loss would mean something different per dataset.
    The mean over channels keeps one voxel worth the same whatever it was asked.

    `label_smoothing` divides by two rather than by `n_class` for the same reason: each
    channel is a two-outcome question, and smoothing it towards `n_class` outcomes would pull
    a binary target towards a number that is not in it.
    """
    if label_smoothing > 0:
        target = target * (1 - label_smoothing) + label_smoothing / 2
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def binary_focal(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """Focal loss per channel, where the easy answer may be "no" rather than "background".

    `alpha` means something different from its categorical namesake and there is no way to
    make it mean the same thing. There, it weights the foreground classes against the
    background *channel*, which nested regions do not have. Here it weights the positive
    answer against the negative one **within each channel** - `alpha` on the voxels in the
    region, `1 - alpha` on those outside it - which is the original two-class formulation and
    the only reading available once no channel is background.
    """
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits)
    # The probability assigned to whichever answer is true, without re-exponentiating a
    # sigmoid: pt = p where target is 1, 1 - p where it is 0.
    log_pt = target * log_positive + (1 - target) * log_negative
    weight = (1 - log_pt.exp()).clamp_min(0).pow(gamma)
    if alpha is not None:
        weight = weight * (target * alpha + (1 - target) * (1 - alpha))
    return -(weight * log_pt).mean()


def combo(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    ce_ratio: float = 0.5,
    smooth: float = 1.0,
    *,
    multilabel: bool = False,
) -> torch.Tensor:
    """Taghanaki's Combo loss: a lopsided cross-entropy blended with Dice.

    `ce_ratio` tilts the cross-entropy between punishing false negatives (towards 1) and
    false positives (towards 0); `alpha` sets how much of the total each half provides.

    Its cross-entropy half is already written out per channel rather than through
    `cross_entropy`, so `multilabel` only has to reach the probabilities: with a sigmoid the
    two terms stop being one voxel's two halves and become each channel's own question,
    which is what the formula was doing to the background channel anyway.
    """
    probs = probabilities(logits, multilabel=multilabel).clamp(EPS, 1 - EPS)
    weighted_ce = -(
        ce_ratio * target * probs.log() + (1 - ce_ratio) * (1 - target) * (1 - probs).log()
    ).mean()
    dice_loss = 1 - dice_coefficient(probs, target, smooth).mean()
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


# ------------------------------------------------------------------ regression
# These three are the odd family here, in a way worth stating before the code: **nothing
# activates their input.** Everything above turns logits into probabilities first, because
# an overlap only means something between 0 and 1; a flow field or a distance in pixels is
# unbounded, and putting a sigmoid in front of one would squash it into (0, 1) while still
# producing a number and a gradient - §4's worst available outcome. So `prediction` here is
# the model's raw output used as the prediction, and the word "logits" does not apply.


def masked_mean(error: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of `error` over the voxels `mask` selects, or over all of them if it is None.

    The mean is **per selected voxel, not per voxel of the picture**: dividing by the mask's
    own total is what makes the number "average error where the target is defined" rather
    than a quantity that shrinks as the object does. A mask of (B, 1, *spatial) is expanded
    across channels, so one foreground selects every component of a vector target.

    An empty mask gives exactly 0 - there is nothing to average - and the clamp is what
    keeps that from being 0/0. That is the honest answer for a tile with no object in it,
    and it is worth knowing that such a tile then contributes no gradient at all: on this
    data 167 of 536 Data Science Bowl cases are under 5% foreground, so with small tiles
    the empty ones are not a corner case.
    """
    if mask is None:
        return error.mean()
    if mask.shape[0] != error.shape[0] or mask.shape[2:] != error.shape[2:]:
        raise ValueError(
            f"the mask's shape {tuple(mask.shape)} does not match the target's "
            f"{tuple(error.shape)} in batch or space"
        )
    weights = mask.expand_as(error)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def mse(
    prediction: torch.Tensor, target: torch.Tensor, *, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Squared error, averaged where the mask says the target means something."""
    return masked_mean(F.mse_loss(prediction, target, reduction="none"), mask)


def l1(
    prediction: torch.Tensor, target: torch.Tensor, *, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Absolute error, averaged where the mask says the target means something."""
    return masked_mean(F.l1_loss(prediction, target, reduction="none"), mask)


def smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared inside `beta` of the truth and absolute outside it.

    The gradient of `mse` grows without bound in the error, so one badly wrong voxel can
    dominate a batch; `l1`'s gradient is constant, which makes it indifferent to being
    nearly right. This is the usual compromise, and `beta` is where it switches.
    """
    return masked_mean(F.smooth_l1_loss(prediction, target, reduction="none", beta=beta), mask)
