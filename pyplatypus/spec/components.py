"""Losses, metrics, optimisers, callbacks, augmentation.

Each family is a discriminated union keyed on `name`. The old package typed these as
`Any`, which meant a misspelled loss or a nonsense parameter sailed through validation
and failed much later, deep inside training. Here `{"name": "focal", "gamm": 1}` fails
immediately and says which key it did not recognise.

The YAML for one of these reads:

    loss:
      name: focal
      gamma: 2.0
    metrics:
      - name: iou
      - name: tversky
        alpha: 0.7
"""

from __future__ import annotations

from difflib import get_close_matches
from functools import lru_cache
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from pyplatypus.spec.common import SpecModel

# --------------------------------------------------------------------------- losses
# Every loss here reduces over "all dimensions except batch and channel", which is what
# lets the same implementation serve 2D and 3D untouched (PLAN.md, rule 3).


# `smooth` means the same thing in every loss that has it, so one home. `alpha` does not -
# it is a class weight in focal, the weight on false negatives in Tversky, the mix in combo
# and the surface term's share in boundary - so each of those says its own thing. A field is
# identified by (model, name), never by name.
_SMOOTH = (
    "Added to both numerator and denominator so a class absent from an image gives a finite "
    "number instead of 0/0. Leave it alone unless you know why you are moving it: raising it "
    "makes an empty class look better than it is."
)
_LOSS_NAME = (
    "Which loss, and it decides which other keys this block accepts: `tversky` takes "
    "`alpha`, `focal` takes `gamma`, and giving a key the chosen loss does not have is "
    "refused by name rather than ignored."
)
_OPTIMIZER_NAME = (
    "Which optimiser, and it decides which other keys this block accepts - `sgd` takes "
    "`momentum`, the Adam family takes `beta_1` and `beta_2`. `learning_rate` and "
    "`weight_decay` are accepted by all of them."
)


class _Loss(SpecModel):
    #: Whether this loss means anything when a voxel may be in several classes at once -
    #: `regions`, where each channel is an independent question rather than one of a set of
    #: alternatives. True for every loss here but `lovasz`, and it is a property of the
    #: specification rather than of the module because the refusal belongs before anything
    #: is built. `pyplatypus.objectives.losses` reads this rather than keeping its own list.
    supports_multilabel: ClassVar[bool] = True

    #: Whether the target this scores is continuous rather than a set of classes. The same
    #: shape of claim as the flag above and for the same reason: `instance_segmentation`
    #: has to refuse `loss: mse` over a one-hot target, and a flow block's own
    #: `loss: dice` over a vector field, by name and before anything is built - and a list
    #: of names kept in the instance spec would be a second place for the answer to live.
    #: Read by `InstanceModel` and by `Flows`, between them the only things that have both
    #: kinds of channel at once.
    continuous: ClassVar[bool] = False

    #: Whether this loss is defined over a *set* of elements rather than over each element
    #: on its own. An overlap is a ratio of areas and a boundary term is an integral along
    #: one, so both need something with extent; `lovasz` ranks the per-element errors, which
    #: needs several of them to rank. One label for a whole image has none of that, which is
    #: why `ClassificationModel` reads this flag rather than keeping a list of names - the
    #: third claim of this shape in this class, after the two above.
    #:
    #: **Declared rather than measured, and the difference is itself measured.** A loss that
    #: says `False` here must give the same number for one element as for four identical
    #: ones, and `tests/objectives/test_extent.py` asserts exactly that. The converse does
    #: not hold: `lovasz` is extent-invariant in value - 0.382946 either way - while its
    #: mechanism is vacuous on one element, so the guard runs in one direction only and the
    #: exception is pinned rather than left to be rediscovered.
    needs_extent: ClassVar[bool] = True


class IouLoss(_Loss):
    """Dice's harsher twin: the same ordering, a lower number.

    Without smoothing the coefficient is exactly `dice / (2 - dice)`, which is strictly
    increasing - measured here over twelve predictions, the two rank them identically. So
    the choice is about which number you want to read, not which model you get. As a loss
    the gradients still differ, and whether that changes a trained model is not measured.
    """

    name: Literal["iou"] = Field("iou", description=_LOSS_NAME)
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)


class DiceLoss(_Loss):
    """Overlap, and nothing about where the error sits.

    The one to reach for first on a single foreground class. It counts a wrong voxel the
    same whether it sits in the middle of the object or a long way outside it, which is
    `boundary`'s whole reason for existing, and on a small object a single missed piece
    moves it a lot. `smooth` is what keeps it from 0/0 on an empty mask.
    """

    name: Literal["dice"] = Field("dice", description=_LOSS_NAME)
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)


class GeneralizedDiceLoss(_Loss):
    """Dice with every class weighted by the inverse square of how much of the picture it
    occupies, so a rare class counts for more than an equal share rather than for less.

    **The usual argument for this loss does not apply here and is not made.** It is
    normally introduced as a fix for Dice being dominated by the background; `dice` in this
    package is a mean over the class channels, so a lesion in 1% of the frame already
    carries the same weight as the 99% around it. Measured on a 1% target with two
    predictions - one finding the lesion and leaking a false positive of equal size, one
    missing it entirely and getting the background perfect:

        dice               0.1658 finding   0.4731 missing   prefers finding by 0.3073
        cce                0.1600 finding   0.1600 missing   indifferent
        cce_dice           0.1629 finding   0.3166 missing   prefers finding by 0.1536
        generalized_dice   0.3278 finding   0.9511 missing   prefers finding by 0.6233

    So what this adds is weight **beyond** an equal share: on that target the classes are
    weighted 1 to 9801 rather than 1 to 1, and the margin between finding the lesion and
    missing it doubles. And the objective that is genuinely dominated is the per-pixel half
    of `cce_dice`: to `cce` a false positive and a missed lesion of the same size are the
    same number of wrong pixels, which is the indifference in that table.

    **The run that motivates it is in this project's record and the loss was never the
    variable.** On SIIM-ACR pneumothorax, a median 0.59% of the frame, the built-in encoder
    scored `0.0531 +/- 0.0586` over three seeds - a spread larger than its own mean, with
    two of three seeds never learning at all - and every arm of that table used `cce_dice`.

    **Not a third `tversky`.** `tversky` moves the balance between false positives and
    false negatives by a number you choose; `focal_tversky` reweights hard examples. This
    reweights **classes by how much of the image they occupy**, computed from the batch
    rather than set by hand. Three losses, three questions, and a reader who picks by name
    will pick wrongly.

    Two things to know before reaching for it. With every class the same size it is
    **exactly `dice` pooled over the classes** - a test rather than a claim - so it costs
    nothing and buys nothing on a balanced problem. And the weights are recomputed per
    batch, so a batch holding no lesion at all is scored on flat weights; with `batch_size`
    1 on a sparse dataset that is most batches, which is an argument for a larger batch
    rather than against the loss.
    """

    name: Literal["generalized_dice"] = Field("generalized_dice", description=_LOSS_NAME)
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)


class CceLoss(_Loss):
    """Per-pixel classification, with no notion of the object.

    Healthy gradients from the first epoch, which the overlap losses do not have while
    everything is still wrong. The cost is that it optimises what is numerous: with a
    background of 95% of the voxels, which in medical images is ordinary, it is mostly
    learning the background. That is what `cce_dice` exists to fix.
    """

    needs_extent: ClassVar[bool] = False

    name: Literal["cce"] = Field("cce", description=_LOSS_NAME)
    label_smoothing: float = Field(
        0.0,
        ge=0,
        lt=1,
        description=(
            "Train towards 1 - this value instead of towards 1, so the model is not pushed "
            "to be certain. Worth having where the labels themselves are uncertain, which on "
            "a hand-drawn boundary they are."
        ),
    )


class CceDiceLoss(_Loss):
    """The blend to start a segmentation run with.

    Cross-entropy keeps the gradients healthy while everything is wrong, Dice pulls the
    overlap up once the easy voxels are done. `cce_weight` is the split, and at 0 or 1 it
    is one of the two alone - which is worth knowing before tuning it as if it were a
    third thing.
    """

    name: Literal["cce_dice"] = Field("cce_dice", description=_LOSS_NAME)
    cce_weight: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "How much of the objective is cross-entropy; the rest is Dice. Cross-entropy "
            "cares about every pixel and Dice about the overlap, so raising this helps a "
            "class that covers very little of the image and lowering it favours the shape."
        ),
    )
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)


class FocalLoss(_Loss):
    """Cross-entropy that loses interest in the voxels it already gets right.

    `gamma` is the whole feature: at 0 this is exactly cross-entropy - measured, the two
    agree to floating point - and raising it pushes the loss onto the hard voxels. Worth
    having where the easy background dominates the gradient. The cost is that a model that
    is right but unconfident everywhere is barely trained at all.
    """

    needs_extent: ClassVar[bool] = False

    name: Literal["focal"] = Field("focal", description=_LOSS_NAME)
    gamma: float = Field(
        2.0,
        ge=0,
        description=(
            "How hard to discount the pixels already classified well, so the gradient comes "
            "from the ones that are not. 0 is plain cross-entropy; 2 is the usual value."
        ),
    )
    alpha: float | None = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Weight on the positive class, between 0 and 1. Unset means no weighting, which "
            "is not the same as 0.5: at 0.5 both classes are halved and the gradients are "
            "smaller, which looks like a learning-rate change."
        ),
    )


class TverskyLoss(_Loss):
    """Dice with a thumb on the scale: miss less, or over-segment less.

    `alpha` weights false negatives and `1 - alpha` false positives, so raise it when a
    missed lesion costs more than a flagged healthy voxel. **At 0.5 this is Dice, but at
    twice the smoothing** - measured: `tversky(alpha=0.5, smooth=1)` equals
    `dice(smooth=2)` exactly. Leaving `alpha` at its default is therefore not Tversky; it
    is Dice with a different floor, and that is the thing to notice when comparing runs.
    """

    name: Literal["tversky"] = Field("tversky", description=_LOSS_NAME)
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Weight on false negatives; raising it buys recall at the cost of precision, "
            "which is the trade a screening task usually wants. The weight on false "
            "positives is 1 - alpha and is not a separate key, so the two cannot disagree. "
            "At alpha 0.5 this is Dice exactly, but only when `smooth` is 0."
        ),
    )
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)

    @property
    def beta(self) -> float:
        """Tversky's two weights sum to one; storing both invites them not to."""
        return 1.0 - self.alpha


class FocalTverskyLoss(_Loss):
    """Tversky, with the same loss of interest in what is already right.

    `gamma` raises `1 - TI` to a power, so the cases near the truth stop contributing. At
    `gamma = 1` this is exactly the Tversky loss - measured - so like `focal`, the
    parameter *is* the feature. Reach for it when the asymmetry of `tversky` is wanted and
    a majority of easy cases is drowning it.
    """

    name: Literal["focal_tversky"] = Field("focal_tversky", description=_LOSS_NAME)
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Weight on false negatives, as in `tversky`, with false positives taking 1 - alpha."
        ),
    )
    gamma: float = Field(
        1.0,
        gt=0,
        description=(
            "Exponent on the Tversky index, which is where this differs from `tversky`: "
            "above 1 it concentrates the gradient on the cases scoring badly. Note 1 is the "
            "default and means plain Tversky, not 'off'."
        ),
    )
    smooth: float = Field(1.0, gt=0, description=_SMOOTH)


class ComboLoss(_Loss):
    """A lopsided cross-entropy blended with Dice.

    Two knobs that answer two different questions: `ce_ratio` tilts the cross-entropy
    between punishing missed foreground and punishing false alarms, and `alpha` sets how
    much of the total each half provides. Worth having when both the class imbalance and
    the direction of the error matter; `cce_dice` is the simpler answer when only the
    first does.
    """

    name: Literal["combo"] = Field("combo", description=_LOSS_NAME)
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "How much of the objective is the weighted cross-entropy term; the rest is Dice."
        ),
    )
    ce_ratio: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Within the cross-entropy term, how much weight goes on the positive class. "
            "This is the key that makes `combo` worth choosing over `cce_dice`: it lets the "
            "two halves of the objective disagree about which class matters."
        ),
    )


class LovaszLoss(_Loss):
    """Optimise the overlap you are going to report, rather than a proxy for it.

    A surrogate for the Jaccard index that is differentiable without being smoothed, so it
    acts directly on the quantity `iou` is scored by. It sorts the errors of every image it
    is given, which costs more than the elementwise losses, and `per_image` decides whether
    that happens per image or over the batch: per image is the harsher reading and on mostly
    empty images it is also the noisier one.

    **Mutually exclusive classes only.** The Lovasz-Softmax surrogate sorts the errors of a
    softmax over classes that are alternatives and walks the Lovasz extension of the Jaccard
    loss along that order. The independent form is the Lovasz-Hinge, a different construction
    over signed margins rather than a setting on this one - so `regions` is refused here
    rather than silently served, because run on sigmoid channels this would return a number
    and a gradient, which is the worst available outcome.
    """

    supports_multilabel: ClassVar[bool] = False

    name: Literal["lovasz"] = Field("lovasz", description=_LOSS_NAME)
    per_image: bool = Field(
        False,
        description=(
            "Compute the loss for each image and average, rather than over the whole batch "
            "at once. Per image is the honest thing when images differ in how much of them "
            "is foreground, because otherwise a large object in one image dominates the "
            "batch; it is also noisier."
        ),
    )


#: Everything but `boundary`, which takes one of these as its region term. Spelled out
#: rather than recursive: a boundary loss inside a boundary loss would add two surface
#: terms with no region term holding either of them down, which is the one combination
#: that cannot work.
RegionLossSpec = Annotated[
    IouLoss
    | DiceLoss
    | GeneralizedDiceLoss
    | CceLoss
    | CceDiceLoss
    | FocalLoss
    | TverskyLoss
    | FocalTverskyLoss
    | ComboLoss
    | LovaszLoss,
    Field(discriminator="name"),
]


class BoundaryLoss(_Loss):
    """A region loss plus a term that knows how far a wrong voxel is from the truth.

    Dice and IoU count a voxel the same wherever it sits, which is why a model can reach
    0.88 on Dice while its volumes run a fifth too large: the overshoot is all at the
    boundary, and for a small lesion the boundary is most of the object. This adds
    `mean(phi * p)`, where `phi` is the signed distance to the truth's boundary.

    ```yaml
    loss:
      name: boundary
      region: {name: focal, gamma: 2.0}
      alpha: 0.5
    ```

    `alpha` weights the region term, `1 - alpha` the surface term. It cannot be 1, which
    would leave the surface term doing nothing and the name lying, and it cannot be 0: the
    surface term alone has no notion of how much of the object was found and is minimised
    by a confident prediction deep inside a shrunken one.

    **What it buys and what it costs, measured.** Against Dice alone on lesions with
    ambiguous edges, three seeds, at the default `alpha`: the systematic volume bias goes
    from -5.18% to -0.06%, and the per-case absolute volume error goes the wrong way, from
    16.90% to 22.17%. Dice itself falls by 0.011. So it trades *accuracy per case* for
    *being unbiased over a series* - and which of those you want is the question being
    asked. "Is there a lesion" wants the overlap; "has it grown since March" wants a volume
    that is not systematically wrong, and no single number answers both.

    The bias improvement is suggestive rather than established on three seeds, because the
    baseline's own bias wanders by ±3.11%. The costs are established.

    **It also costs time.** The signed distance transform runs per sample in the loader's
    workers - measured at 5.3 ms for a 256x256 image, 23.4 ms for a 64x64x32 volume and
    709.8 ms at 128^3 - so on large volumes give the loader workers, or the transform
    becomes the training.
    """

    name: Literal["boundary"] = Field("boundary", description=_LOSS_NAME)
    region: RegionLossSpec = Field(
        default_factory=lambda: DiceLoss(),
        description="The overlap term this is added to. Dice unless you say otherwise.",
    )
    alpha: float = Field(
        0.9,
        gt=0,
        lt=1,
        description=(
            "How much of the objective is the region term, `1 - alpha` the surface term.\n\n"
            "**0.9 because 0.5 was measured and is unusable.** Three seeds on lesions with "
            "ambiguous edges, 60 epochs:\n\n"
            "```\n"
            "                 Dice             volume bias       volume |error|\n"
            "dice alone       0.9525 ±0.0027   -5.18% ±3.11%     16.90% ±0.90%\n"
            "boundary a=0.5   0.9164 ±0.0240  +32.22% ±28.28%    42.95% ±19.12%\n"
            "boundary a=0.9   0.9411 ±0.0093   -0.06% ±4.15%     22.17% ±2.88%\n"
            "```\n\n"
            "At 0.5 the surface term has half the objective while the prediction is still "
            "random, which is where it is known to be unstable - the seed spread is six "
            "times the baseline's and every number is worse. Kervadec schedules this "
            "downwards from near 1 rather than fixing it; 0.9 is the closest a single "
            "number comes, and the spec deliberately has no schedule to hide behind.\n\n"
            "Excludes both ends: at 1 the surface term is absent and the name is a lie, "
            "and at 0 nothing measures how much of the object was found."
        ),
    )


# --------------------------------------------------------------- regression losses
# The only family here that scores a continuous target rather than an overlap, and the
# only one whose input is used unactivated. They exist for `flows` and `hv` - a continuous
# instance target has nothing in the ten overlap-and-cross-entropy losses to be trained
# against - and `where` is their whole content. `pyplatypus#209`.
#
# **Nameable only where there is a continuous target**, which is a `separation: flows`
# block's own `loss` and nowhere else. A model's own `loss` scores the classes at every
# segmentation task, so `SegmentationModel.the_mask_needs_a_loss_over_classes` refuses
# these there by name - and `available_losses()` leaves them out of the semantic task's
# list for the same reason. Both read `continuous` below rather than these three names,
# so a fourth regression loss is refused and unlisted with nothing else edited.
# `pyplatypus#223`.
#
# `supports_multilabel` stays at the inherited True, and the honest reading is that the
# question does not arise: nothing here turns an output into a probability, so there is no
# softmax-against-sigmoid decision to get wrong.
_WHERE = (
    "Where the target means anything, and so where the error is averaged.\n\n"
    "A continuous instance target is defined on the objects and undefined off them - there "
    "is no direction towards the centre for a pixel in no object - so the builder writes "
    "zero on the background. Measured on the 536 Data Science Bowl cases: the foreground is "
    "**13.97% of the pixels at 256x256** (13.48% at native size, 0.03% to 59.2% per case, "
    "167 cases under 5%), so `everywhere` spends 86% of the objective on a value a single "
    "output bias satisfies exactly.\n\n"
    "`foreground` divides by the mask's own total, which makes the number mean *average "
    "error per object voxel*. `everywhere` is the other defensible answer and is what "
    "CellPose and HoVer-Net's plain regression terms do - their background zero is a real "
    "value rather than a missing one. Both are offered and neither is guessed; `foreground` "
    "is the default because it is the reading whose number means something.\n\n"
    "`foreground` asks the loader for the mask, which is the same mechanism `boundary` uses "
    "for its distance map. It is never read off the target: a flow field is exactly zero at "
    "every object's centre, so a mask taken from the non-zero voxels would drop precisely "
    "the seeds - and the seeds are what limits this approach (`pyplatypus#211`)."
)


class _Regression(_Loss):
    continuous: ClassVar[bool] = True
    #: A squared or absolute error is a mean over elements, so one element is a legitimate
    #: case arithmetically. These are refused for a classification label by `continuous`
    #: instead, which is the question that actually disqualifies them.
    needs_extent: ClassVar[bool] = False

    where: Literal["foreground", "everywhere"] = Field("foreground", description=_WHERE)


class MseLoss(_Regression):
    """Squared error on a continuous target.

    For `flows` and `hv`, and for nothing a mask can do: run against a one-hot target this
    scores raw output against 0 and 1 with no activation in between, which is not a
    segmentation objective. The gradient grows with the error, so one badly wrong voxel
    counts for a lot - right when the large values of the target are the informative ones,
    wrong when the labels are noisy.

    ```yaml
    loss:
      name: mse
      where: foreground
    ```
    """

    name: Literal["mse"] = Field("mse", description=_LOSS_NAME)


class L1Loss(_Regression):
    """Absolute error on a continuous target.

    The same gradient however wrong the voxel is, which makes it robust to a handful of bad
    labels and indifferent to being nearly right. Reach for it where the target has
    outliers you do not want the model chasing.
    """

    name: Literal["l1"] = Field("l1", description=_LOSS_NAME)


class SmoothL1Loss(_Regression):
    """`mse` close to the truth and `l1` far from it, switching at `beta`.

    The usual compromise, and the one the detector already uses on box size. At a very small
    `beta` this is `l1` and at a very large one it is `mse` scaled by `beta`, which is worth
    knowing before tuning it as if it were a third thing.
    """

    name: Literal["smooth_l1"] = Field("smooth_l1", description=_LOSS_NAME)
    beta: float = Field(
        1.0,
        gt=0,
        description=(
            "Where the quadratic part ends and the linear one begins, in the target's own "
            "units. For a flow field scaled to roughly [-1, 1] the default makes the whole "
            "range quadratic, so lower it to get any of the robustness this loss is for."
        ),
    )


LossSpec = Annotated[
    IouLoss
    | DiceLoss
    | GeneralizedDiceLoss
    | CceLoss
    | CceDiceLoss
    | FocalLoss
    | TverskyLoss
    | FocalTverskyLoss
    | ComboLoss
    | LovaszLoss
    | BoundaryLoss
    | MseLoss
    | L1Loss
    | SmoothL1Loss,
    Field(discriminator="name"),
]

# -------------------------------------------------------------------------- metrics


class _Metric(SpecModel):
    #: Whether this measures a property of a whole mask - a skeleton, a boundary - rather
    #: than an overlap count, and therefore cannot be rebuilt from the pieces of a tiled
    #: image. The same shape of claim as `continuous` on a loss and for the same reason:
    #: a tiled run's epochs cannot honestly report it, so a callback asking to watch it is
    #: refused when the configuration is read, and that refusal lives in the specification
    #: layer where there is no built metric to ask. The built metric answers the same
    #: question as `accumulates`, and a test pairs the two - this flag was a list of names
    #: in `spec.models` for one release, and `surface_dice` was added to one side of it.
    whole_mask: ClassVar[bool] = False


class _Overlap(_Metric):
    smooth: float = Field(
        1.0,
        ge=0,
        description=(
            "Zero is allowed here, unlike in the losses. Smoothing inflates a reported "
            "score: a class absent from an image scores a perfect 1.0 with smooth>0. "
            "Use 0 for an honest number, at the cost of NaN on absent classes."
        ),
    )
    include_background: bool = Field(
        True,
        description=(
            "Average over every class, background included. In medical images the "
            "background is often 95% of the pixels, so leaving it in flatters the score; "
            "papers usually report foreground only. Set false for that."
        ),
    )


_METRIC_NAME = (
    "Which metric. These are computed on the mask rather than on the objective, so they "
    "stay comparable between models trained on different losses."
)


class IouMetric(_Overlap):
    """The same ordering as Dice, on the scale papers outside medicine use.

    Without smoothing it is exactly `dice / (2 - dice)`, measured here, so it adds no
    information to a report that already carries Dice - it is the convention of the
    audience rather than a second opinion.
    """

    name: Literal["iou"] = Field("iou", description=_METRIC_NAME)


class DiceMetric(_Overlap):
    """The number to report unless there is a reason not to.

    Scored on the mask rather than on the objective, so a Dice-trained and a focal-trained
    model are comparable. It ranks models identically to `iou`, so reporting both is one
    result twice; `smooth` and `include_background` are what make it flatter or honest.
    """

    name: Literal["dice"] = Field("dice", description=_METRIC_NAME)


class TverskyMetric(_Overlap):
    """A score that can be asked to care more about what was missed.

    The metric form of the loss, and useful for reporting a recall-leaning number beside
    the symmetric one. Moving `alpha` is the only reason to choose it: at the default it is
    Dice, with the same doubling of the smoothing the loss has.
    """

    name: Literal["tversky"] = Field("tversky", description=_METRIC_NAME)
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Weight on false negatives, with 1 - alpha on false positives - the same meaning "
            "as in the Tversky loss. At 0.5 this is Dice, so it is worth moving or not "
            "reporting at all: a Tversky metric at 0.5 beside a Dice metric is one number "
            "twice."
        ),
    )


class ClDiceMetric(_Overlap):
    """Whether the structure is connected, rather than whether the voxels coincide.

    The one metric here that measures a property of the whole mask: it scores the overlap
    of the prediction's skeleton with the truth's, so a vessel broken into two pieces loses
    where Dice barely notices. Measured on a constructed pair - a vessel drawn two voxels
    thinner, against the same vessel cut in half - Dice reads 0.8750 and 0.9474 and so
    *prefers* the severed one, while this reads 1.0000 and 0.9425 and does not. Both pairs
    are pinned by a test.

    **What it was worth on a real model**, FIVES, 200 cases: correlation 0.9779 with Dice,
    mean difference +0.0041, 127 cases higher and 73 lower. At that quality it is not
    telling a different story - which is itself the result, and the reason to read both.

    Not a drop-in. A one-voxel structure displaced by one voxel scores 0, the skeleton
    needs a whole mask so a tiled run reports it per case through `evaluate_cases`, and it
    is absent from the per-epoch history for the same reason.
    """

    whole_mask: ClassVar[bool] = True

    name: Literal["cldice"] = Field(
        "cldice",
        description=(
            "Centreline Dice, which measures whether a structure is connected rather than "
            "whether its pixels coincide. A skeleton is a property of a whole mask, so on a "
            "tiled run (`splits`) it is reported **per case** by `evaluate_cases`, which "
            "reassembles a case's tiles before measuring it. It is absent from the epoch "
            "and from the history there, because a per-batch average of a per-tile skeleton "
            "would be a number about nothing - which also means a callback cannot watch "
            "`val_cldice` on a tiled run, and the specification says so when it is read."
        ),
    )
    include_background: bool = Field(
        False,
        description=(
            "False here where the other metrics default to True, and not for the usual "
            "reason. The background's skeleton lies inside the background by construction, "
            "so that class scores 1.0 whatever the model did: measured on a 3-pixel vessel, "
            "averaging it in moved 0.7865 to 0.8933. It is not a flattering average, it is "
            "a number with no meaning in it."
        ),
    )
    iterations: int = Field(
        5,
        ge=1,
        le=50,
        description=(
            "How many times the skeleton is peeled. It has to reach the half-width of the "
            "thickest structure or its core is never reduced to a centreline, and five "
            "covers a ten-pixel object. A retinal vessel is about nine pixels wide in the "
            "data - measured on FIVES as mask area over skeleton length, not the one to "
            "five often quoted, which is what a naive resize leaves behind - so five is "
            "enough for vessels and not much more than enough. Raise it for something "
            "thicker, at a cost linear in the number; too few is reported rather than "
            "scored, because an empty skeleton otherwise reads as a perfect 1.0."
        ),
    )


class SurfaceDiceMetric(_Overlap):
    """How much of each boundary lies within a tolerance of the other, rather than how much
    of the volume coincides.

    Dice asks about volume. `cldice` asks about connectedness. Neither asks **how far
    wrong** a boundary is, and that is what is asked of a contour somebody has to approve:
    the volume vignette ended on a model at Dice 0.88 whose volumes were 12% to 23% too
    large, and this is the number that says so directly rather than being inferred from two
    columns that disagree. It is also what `boundary` optimises and nothing here reported,
    so that loss was being judged by the metric it exists to improve on.

    Normalised surface Dice as published (Nikolov et al., 2018), and symmetric on purpose:
    the truth's surface near the prediction is the boundary's recall, the prediction's
    surface near the truth is its precision, and the single ratio weights the two by
    surface size.

    **The one metric here that reads the voxel spacing**, which is why it belongs in a
    package that carries one. With `target_spacing` set every sample is on a common grid and
    `tau` is in millimetres; without it each sample was resized independently and `tau` is
    in voxels - the same pair of answers the signed distance map gives, and for the same
    reason. A 0.5 mm and a 2.5 mm acquisition are not comparable in voxels at all.

    Like `cldice` it is a property of a whole mask, so on a tiled run (`splits`) it is
    reported per case by `evaluate_cases` rather than per batch.
    """

    whole_mask: ClassVar[bool] = True

    name: Literal["surface_dice"] = Field(
        "surface_dice",
        description=(
            "Normalised surface Dice: the fraction of each boundary lying within `tau` of "
            "the other. A surface is a property of a whole mask, so on a tiled run "
            "(`splits`) it is reported **per case** by `evaluate_cases`, which reassembles "
            "a case's tiles first, and it is absent from the epoch history - a per-batch "
            "average of per-tile boundaries would be a number about nothing."
        ),
    )
    tau: float = Field(
        ...,
        gt=0,
        description=(
            "The tolerance, and it is **required because no default is honest**: a "
            "tolerance belongs to the structure and the dataset, and is conventionally the "
            "inter-observer variability for that structure - the distance two people "
            "drawing the same boundary differ by. Published values exist per organ and per "
            "challenge; take one from the protocol you are comparing against rather than "
            "from here. In millimetres when `target_spacing` puts the data on a common "
            "grid, and in voxels when it does not."
        ),
    )
    include_background: bool = Field(
        False,
        description=(
            "False here where `dice` and `iou` default to True, and for a measured reason "
            "rather than the usual one. The background's boundary is the foreground's "
            "boundary **plus the edge of the picture**, and that edge agrees trivially "
            "between any two masks - so the background class reads high whatever the model "
            "did. Measured on four random pairs at `tau` 2: background 0.6110 to 0.7030 "
            "against foreground 0.5583 to 0.6266, higher every time, by 0.05 to 0.12. That "
            "is agreement on the field of view rather than on the object."
        ),
    )


MetricSpec = Annotated[
    IouMetric | DiceMetric | TverskyMetric | ClDiceMetric | SurfaceDiceMetric,
    Field(discriminator="name"),
]

# ----------------------------------------------------------------------- optimisers
# torch's set, not TensorFlow's. Ftrl is gone because torch has no Ftrl.


# Descriptions shared between optimisers. `beta_1`, `beta_2` and `eps` mean the same thing
# wherever they appear, so the sentence has one home and each field points at it rather
# than repeating it four times.
_BETA_1 = (
    "Decay for the running average of the gradient. Lower reacts to the last few batches, "
    "higher smooths over more of them; 0.9 is the usual value and rarely worth moving."
)
_BETA_2 = (
    "Decay for the running average of the squared gradient, which is what scales each "
    "parameter's step. Lowering it below about 0.99 makes training noticeably noisier."
)
_EPS = (
    "Added to the denominator so a parameter whose gradient has been zero for a while "
    "does not take an enormous step. It is a guard, not a knob."
)
_MOMENTUM = (
    "Carry a fraction of the previous step into this one. 0 is plain gradient descent; "
    "0.9 is the usual choice and is what makes SGD competitive with the adaptive methods."
)


class _Optimizer(SpecModel):
    learning_rate: float = Field(
        1e-3,
        gt=0,
        description=(
            "How big a step to take. The one setting worth trying first when a run will not "
            "learn: too high and the loss moves without improving, too low and it improves "
            "too slowly to tell from not improving at all. Each parameter group decays from "
            "its own rate, so an `encoder_learning_rate` is not flattened by a schedule."
        ),
    )
    weight_decay: float = Field(
        0.0,
        ge=0,
        description=(
            "Pull weights towards zero at every step, which is L2 regularisation. Off by "
            "default because it interacts with the loss and is not free: on a small dataset "
            "it can cost more than the overfitting it prevents."
        ),
    )


class Adam(_Optimizer):
    """The one to use unless there is a reason not to.

    Adapts the step per parameter, which is why it usually trains without anything being
    tuned except `learning_rate`. **No run in this project has measured one of these eight
    against another on segmentation**, so the honest reading of the list is: this is the
    default, `sgd` with momentum is the literature's recipe when you are prepared to tune a
    schedule, and the rest are here because a configuration should be able to name what
    torch offers.
    """

    name: Literal["adam"] = Field("adam", description=_OPTIMIZER_NAME)
    beta_1: float = Field(0.9, ge=0, lt=1, description=_BETA_1)
    beta_2: float = Field(0.999, ge=0, lt=1, description=_BETA_2)
    eps: float = Field(1e-8, gt=0, description=_EPS)
    amsgrad: bool = Field(
        False,
        description=(
            "Keep the largest squared-gradient average seen so far rather than the current "
            "one, so the step size never grows back. It fixes a convergence case in Adam's "
            "proof and usually changes nothing in practice; worth trying if a run plateaus "
            "and then gets worse."
        ),
    )


class AdamW(_Optimizer):
    """Adam, with weight decay that actually decays the weights.

    Adam applies decay through the gradient, where the adaptive step then rescales it;
    AdamW applies it to the weights directly. That is the entire difference, and it is why
    the default here is 0.01 rather than Adam's 0 - choosing AdamW and leaving the decay at
    zero leaves nothing to choose.
    """

    name: Literal["adamw"] = Field("adamw", description=_OPTIMIZER_NAME)
    beta_1: float = Field(0.9, ge=0, lt=1, description=_BETA_1)
    beta_2: float = Field(0.999, ge=0, lt=1, description=_BETA_2)
    eps: float = Field(1e-8, gt=0, description=_EPS)
    weight_decay: float = Field(
        1e-2,
        ge=0,
        description=(
            "As for the other optimisers, but applied to the weights directly rather than "
            "through the gradient - which is the whole difference between AdamW and Adam, "
            "and why the default here is 0.01 rather than 0."
        ),
    )


class Sgd(_Optimizer):
    """The plain step, and the one that rewards being tuned.

    Slower to get going than the adaptive optimisers and dependent on `learning_rate` and
    usually on a schedule, in exchange for being the recipe most of the classification
    literature's best numbers were produced with. `momentum` is what makes it usable;
    `nesterov` without momentum is refused rather than quietly ignored.
    """

    name: Literal["sgd"] = Field("sgd", description=_OPTIMIZER_NAME)
    momentum: float = Field(0.0, ge=0, description=_MOMENTUM)
    nesterov: bool = Field(
        False,
        description=(
            "Look ahead to where momentum is about to carry the weights before measuring the "
            "gradient. Needs `momentum` above 0, and giving it without momentum is refused "
            "rather than ignored."
        ),
    )

    @model_validator(mode="after")
    def nesterov_needs_momentum(self):
        if self.nesterov and self.momentum == 0:
            raise ValueError("nesterov requires momentum greater than 0")
        return self


class RmsProp(_Optimizer):
    """Adam's denominator without Adam's momentum.

    Divides the step by a running average of squared gradients - `alpha` is this family's
    name for Adam's `beta_2` - and leaves the direction to the raw gradient unless
    `momentum` is set. Worth trying where Adam's first moment seems to be carrying the run
    past a minimum.
    """

    name: Literal["rmsprop"] = Field("rmsprop", description=_OPTIMIZER_NAME)
    alpha: float = Field(
        0.99,
        ge=0,
        lt=1,
        description=(
            "Decay for the running average of the squared gradient - RmsProp's name for what "
            "Adam calls `beta_2`, and it does the same work."
        ),
    )
    momentum: float = Field(0.0, ge=0, description=_MOMENTUM)
    eps: float = Field(1e-8, gt=0, description=_EPS)


class Adagrad(_Optimizer):
    """Accumulates every squared gradient it has ever seen, and so slows to a stop.

    Each parameter's step shrinks monotonically, which suits features that appear rarely
    and does not suit images, where every voxel contributes at every step. On a long run it
    can stop learning without the loss ever looking wrong; `adadelta` is the same idea with
    a window instead of a sum.
    """

    name: Literal["adagrad"] = Field("adagrad", description=_OPTIMIZER_NAME)
    lr_decay: float = Field(
        0.0,
        ge=0,
        description=(
            "Shrink the rate further as training goes on, on top of the shrinking Adagrad "
            "already does by accumulating every squared gradient it has seen. Adagrad's step "
            "only ever gets smaller, which is why long runs with it can stop learning."
        ),
    )
    eps: float = Field(1e-10, gt=0, description=_EPS)


class Adadelta(_Optimizer):
    """Adagrad's idea, bounded by a window instead of a sum.

    `rho` is the length of that window and is the parameter that keeps the step from
    shrinking to nothing. Formulated to need no learning rate at all, though torch still
    takes one and this specification still carries it - so a run that leaves
    `learning_rate` at its default is not using the original method.
    """

    name: Literal["adadelta"] = Field("adadelta", description=_OPTIMIZER_NAME)
    rho: float = Field(
        0.9,
        ge=0,
        lt=1,
        description=(
            "How long a window of squared gradients to remember. Adadelta exists to avoid "
            "Adagrad's ever-shrinking step, and this is the parameter that does it."
        ),
    )
    eps: float = Field(1e-6, gt=0, description=_EPS)


class Adamax(_Optimizer):
    """Adam under the infinity norm, which a few huge batches cannot upset.

    The second moment tracks the largest gradient seen rather than the average of their
    squares, so the step is steadier where a handful of batches carry very large gradients
    - a small dataset with one outlier case, for instance. Where none do, it is Adam with
    an extra decision to explain.
    """

    name: Literal["adamax"] = Field("adamax", description=_OPTIMIZER_NAME)
    beta_1: float = Field(0.9, ge=0, lt=1, description=_BETA_1)
    beta_2: float = Field(
        0.999,
        ge=0,
        lt=1,
        description=(
            "As for Adam, except Adamax tracks the largest gradient seen rather than the "
            "average of their squares, which makes it the steadier of the two when a few "
            "batches carry very large gradients."
        ),
    )
    eps: float = Field(1e-8, gt=0, description=_EPS)


class NAdam(_Optimizer):
    """Adam with the look-ahead that `sgd` calls `nesterov`.

    The momentum term is applied before the gradient is measured rather than after. It
    tends to move faster in the first epochs and to land in the same place; worth trying
    where a run is slow to leave its initialisation.
    """

    name: Literal["nadam"] = Field("nadam", description=_OPTIMIZER_NAME)
    beta_1: float = Field(0.9, ge=0, lt=1, description=_BETA_1)
    beta_2: float = Field(0.999, ge=0, lt=1, description=_BETA_2)
    eps: float = Field(1e-8, gt=0, description=_EPS)


OptimizerSpec = Annotated[
    Adam | AdamW | Sgd | RmsProp | Adagrad | Adadelta | Adamax | NAdam,
    Field(discriminator="name"),
]

# ------------------------------------------------------------------------ callbacks
# torch ships no callbacks; these are our own, specified now and implemented in step 5.


class _Callback(SpecModel):
    pass


_CALLBACK_NAME = (
    "Which callback, and it decides which other keys this block accepts. Giving a key the "
    "chosen callback does not have is refused by name rather than ignored."
)
_MONITOR = (
    "What to watch: `val_loss`, `train_loss`, or `val_` followed by a metric the model asks "
    "for, such as `val_dice`. Whether the number should rise or fall is worked out from its "
    "name - anything ending in `loss` is minimised. Watching something no metric will "
    "produce is refused when the specification is built, and so is watching `val_*` in a "
    "run with `validation: false`, rather than waiting for a number that never arrives."
)
_PATIENCE = "Epochs to wait, with no improvement, before acting."
#: The pair that says when a callback begins to act. Two fields rather than one that takes
#: either, because `1` and `1.0` are the same value once YAML is parsed - so "epoch one" and
#: "the whole run" would be one configuration, and the mistake would be silent, since both
#: are in range for both meanings.
_WARMUP_START = (
    "Do not act before this fraction of the run has passed - 0.1 leaves the first tenth "
    "alone. Mutually exclusive with `start_epoch`: a fraction survives a change to "
    "`epochs` and an epoch count does not."
)
_WARMUP_EPOCH = (
    "Do not act before this epoch, 1-based. Mutually exclusive with `start`. This is the "
    "one to use when the warm-up is a property of the optimisation rather than of the "
    "run's length - a schedule's warm-up is fifteen epochs whether the run is 60 or 600."
)


class EarlyStopping(_Callback):
    """Stop when the watched number stops improving, and keep the weights that did.

    What it buys is the epochs after the best one, which on a long run is most of the wall
    clock. The failure to watch for is stopping on noise: where the monitored number
    wanders, `patience` alone is not enough and `min_delta` is what makes an improvement
    count. `restore_best` is on, because the last weights are by definition the ones that
    were not improving.
    """

    name: Literal["early_stopping"] = Field("early_stopping", description=_CALLBACK_NAME)
    monitor: str = Field("val_loss", description=_MONITOR)
    patience: int = Field(10, ge=1, description=_PATIENCE)
    min_delta: float = Field(
        0.0,
        ge=0,
        description=(
            "Improvement smaller than this does not count as improvement. Useful where the "
            "watched number wanders: without it, noise alone keeps resetting the patience."
        ),
    )
    restore_best: bool = Field(
        True,
        description=(
            "Put the best weights back when training stops, rather than keeping the last - "
            "which are by definition the ones that were not improving. Note that `swa` "
            "replaces nothing if a run stops before averaging began."
        ),
    )
    start: float | None = Field(
        None,
        gt=0,
        lt=1,
        description=(
            f"{_WARMUP_START} Unset is no warm-up, which is what this callback did "
            "before `0.8.0a34`.\n\n"
            "**It delays the stopping and nothing else.** An epoch before the warm-up ends "
            "can still become the best one and `restore_best` can still return to it - "
            "those weights are the best the run has if it ends for some other reason, and "
            "throwing them away to tidy the bookkeeping would lose a real answer."
        ),
    )
    start_epoch: int | None = Field(
        None,
        ge=1,
        description=_WARMUP_EPOCH,
    )

    @model_validator(mode="after")
    def one_way_of_saying_when_to_begin(self):
        if self.start is not None and self.start_epoch is not None:
            raise ValueError(
                "early_stopping takes `start` or `start_epoch`, not both: `start` is a "
                "fraction of the run and `start_epoch` is an epoch number."
            )
        return self


class ModelCheckpoint(_Callback):
    """The run's safety net - one file, overwritten.

    Not a history and not a publication: `export_weights()` is what writes a file somebody
    else can load, with a sidecar saying what it is. Use this so an interrupted run is not
    a lost run, and `save_best_only: false` when interruption rather than the best epoch is
    what you are guarding against.
    """

    name: Literal["model_checkpoint"] = Field("model_checkpoint", description=_CALLBACK_NAME)
    path: str = Field(
        description=(
            "Where to write the checkpoint. One file, overwritten - this is the run's safety "
            "net, not a history; `export_weights()` is what produces a file to publish, with "
            "a sidecar recording what it is. **The extension decides the format**: "
            "`.safetensors` is written as safetensors and anything else as a torch archive, "
            "which is what `.pt` has always meant here. Reading does not depend on the name - "
            "both formats are recognised by their contents - so a file written by an older "
            "version still loads under whatever it was called."
        ),
    )
    monitor: str = Field("val_loss", description=_MONITOR)
    save_best_only: bool = Field(
        True,
        description=(
            "Write only when the watched number improves. With `false` every epoch is "
            "written, which is what to use when a run may be interrupted rather than when "
            "the best epoch is wanted."
        ),
    )
    when: dict[str, float] | None = Field(
        None,
        description=(
            "Thresholds that must all hold before anything is written: "
            "`{val_dice: 0.9, val_loss: 0.2}` writes nothing until the model is at least "
            "that good. Each key is a quantity this model reports, checked when the "
            "specification is built rather than at the first epoch.\n\n"
            "**The direction is derived, not configured.** Anything ending in `loss` has "
            "to be *below* its threshold and everything else *above* it, which is the same "
            "rule `monitor` already uses - so there is no `above`/`below` to get backwards."
            "\n\n"
            "**This is an extra gate, not a replacement for `save_best_only`.** With both, "
            "a file is written when the watched number improves *and* every threshold "
            "holds.\n\n"
            "**A threshold is a number about your data.** Unlike `patience` or `factor` it "
            "does not transfer: `val_dice: 0.9` copied from somebody else's configuration "
            "means something entirely different on other images. And a run whose "
            "conditions never hold writes nothing at all - the callback says so when the "
            "run ends, rather than leaving an empty directory to be discovered later."
        ),
    )


class ReduceLrOnPlateau(_Callback):
    """Cut the learning rate when the evidence says the run has stalled.

    The difference from `cosine_annealing` is that this one reacts rather than follows a
    schedule, so it may never fire at all - which is why a cosine beside `swa` is refused
    and this is not. Pairs with a run whose plateau arrives at an epoch nobody can predict.
    """

    name: Literal["reduce_lr_on_plateau"] = Field(
        "reduce_lr_on_plateau", description=_CALLBACK_NAME
    )
    monitor: str = Field("val_loss", description=_MONITOR)
    factor: float = Field(
        0.1,
        gt=0,
        lt=1,
        description="Multiply the learning rate by this when the watched number stops moving.",
    )
    patience: int = Field(5, ge=1, description=_PATIENCE)
    min_lr: float = Field(
        0.0,
        ge=0,
        description=(
            "The floor the rate will not go below. Unlike `cosine_annealing`, this one acts "
            "on evidence rather than on a schedule, so it may never fire at all - which is "
            "why it is not refused alongside `swa` and a cosine is."
        ),
    )


class CosineAnnealing(_Callback):
    """Decay the learning rate along a cosine, from its initial value to `min_lr`.

    No `monitor`: it is a function of how far through the run you are, not of how the run
    is going. That is the difference from `reduce_lr_on_plateau`, and the two **cannot be
    combined** - measured rather than assumed. A schedule recomputes the rate from the
    optimiser's initial value at the end of every epoch, so it erases whatever the plateau
    callback cut: with both configured the rate follows the cosine exactly, 0.0976, 0.0905,
    0.0794 from 0.1, where the plateau alone reaches 1e-6 over the same six epochs. This
    sentence used to say a run may want both, which was never true of the code.
    """

    name: Literal["cosine_annealing"] = Field("cosine_annealing", description=_CALLBACK_NAME)
    min_lr: float = Field(
        0.0,
        ge=0,
        description=(
            "The rate at the last epoch, which the cosine decays to from the optimiser's own "
            "learning_rate. Each parameter group decays from its own initial rate, so an "
            "encoder_learning_rate is not flattened at the first epoch - that would have been "
            "invisible, since the history records one learning_rate, the first group's."
        ),
    )
    epochs: int | None = Field(
        None,
        ge=1,
        description=(
            "How many epochs to spread the decay over. Unset means the model's `epochs`, "
            "which is what you want: a cosine that ends where the run ends. Setting it "
            "shorter parks the rate at `min_lr` for the remaining epochs, and longer "
            "stops the run part way down the curve."
        ),
    )


class LinearAnnealing(_Callback):
    """Decay the learning rate in a straight line, from its initial value to `min_lr`.

    `cosine_annealing`'s simpler sibling and the same shape of argument: a function of how
    far through the run you are rather than of how the run is going. A straight line spends
    longer at high rates than a cosine does at the start and less at the end, which is the
    whole difference; which one is better is a property of the problem and not a thing this
    package has measured.

    Like every schedule here, it is exclusive with the other schedules and with
    `reduce_lr_on_plateau`.
    """

    name: Literal["linear_annealing"] = Field("linear_annealing", description=_CALLBACK_NAME)
    min_lr: float = Field(
        0.0,
        ge=0,
        description=(
            "The rate at the last epoch, which the line descends to from the optimiser's own "
            "learning_rate. Each parameter group descends from its own initial rate, so an "
            "encoder_learning_rate is not flattened at the first epoch."
        ),
    )
    epochs: int | None = Field(
        None,
        ge=1,
        description=(
            "How many epochs to spread the decay over. Unset means the model's `epochs`, "
            "which is what you want: a line that reaches `min_lr` where the run ends. "
            "Setting it shorter parks the rate at `min_lr` for the remaining epochs."
        ),
    )


class StepDecay(_Callback):
    """Multiply the learning rate by `factor` at fixed points in the run.

    The schedule most published recipes are written in - *divide by ten at epochs 30 and
    60* - which is why it is here beside the curves: reproducing somebody's result should
    not require translating their staircase into a cosine.

    The rate is computed from the optimiser's initial value and the number of drops so far
    rather than by repeatedly multiplying what is there, so it does not drift and it is the
    same whether an epoch is replayed or not.
    """

    name: Literal["step_decay"] = Field("step_decay", description=_CALLBACK_NAME)
    factor: float = Field(
        0.1,
        gt=0,
        lt=1,
        description=(
            "What the rate is multiplied by at each drop. 0.1 is the usual recipe's divide "
            "by ten. It must be below 1: a step schedule that raises the rate is a different "
            "idea with different arguments, and silently accepting it here would hide a typo."
        ),
    )
    every: int | None = Field(
        None,
        ge=1,
        description=(
            "Drop every this many epochs. Mutually exclusive with `milestones` - one of the "
            "two, never both. A period survives a change to `epochs`, where a list of "
            "milestones does not, so this is the one to prefer when the schedule is yours "
            "rather than somebody else's."
        ),
    )
    milestones: list[int] | None = Field(
        None,
        description=(
            "The epochs to drop at, 1-based, as a published recipe states them - "
            "`[30, 60]` is divide by ten at thirty and again at sixty. Mutually exclusive "
            "with `every`. Strictly increasing, and an epoch past the end of the run is "
            "refused rather than ignored, because a milestone that never fires is almost "
            "always a number that was meant to fire."
        ),
    )
    min_lr: float = Field(
        0.0,
        ge=0,
        description="The floor the drops stop at, as in `reduce_lr_on_plateau`.",
    )

    @model_validator(mode="after")
    def one_way_of_saying_when(self):
        """`every` or `milestones`, never both and never neither.

        Two fields rather than one that takes either, because `every: 30` and
        `milestones: [30]` are different schedules and a single field holding an integer or
        a list would be read by YAML as whichever the author happened to type.
        """
        if (self.every is None) == (self.milestones is None):
            raise ValueError(
                "step_decay needs exactly one of `every` and `milestones`: `every: 30` "
                "drops every thirty epochs, `milestones: [30, 60]` drops at those two. "
                "Both set is two schedules; neither is none."
            )
        if self.milestones is not None:
            if not self.milestones:
                raise ValueError("step_decay's `milestones` is empty, so it would never drop")
            if any(epoch < 1 for epoch in self.milestones):
                raise ValueError(
                    f"step_decay's `milestones` are 1-based epochs, got {self.milestones}"
                )
            if sorted(set(self.milestones)) != list(self.milestones):
                raise ValueError(
                    f"step_decay's `milestones` must be strictly increasing, got {self.milestones}"
                )
        return self


class Swa(_Callback):
    """Average the weights over the last part of the run instead of taking the final ones.

    Stochastic weight averaging. Past `start`, every epoch's weights are folded into a
    running average and at the end that average becomes the model - the claim being that a
    point in the middle of a flat region generalises better than the corner the last epoch
    stopped in.

    **Batch-normalisation statistics are recomputed afterwards**, by a pass over the
    training data. They have to be: an averaged weight tensor *inherits* the statistics of
    whichever epoch was last rather than averaging them, so without the pass the model is
    evaluated under the wrong normalisation and scores far worse than it should, with
    nothing to say why. It is the step an SWA implementation leaves out.

    **What it is worth, measured.** Three seeds, 60 epochs, lesions with ambiguous edges:

    ```
                 Dice             volume bias        |volume error|
    plain        0.9639 ±0.0100   +1.71% ±12.18%     12.60% ±4.19%
    swa 0.75     0.9667 ±0.0087   +2.08% ±2.15%      12.38% ±4.31%
    swa 0.5      0.9640 ±0.0085   +2.78% ±3.44%      13.75% ±4.21%
    ```

    **Dice does not move and neither does the per-case error** - both are inside the seed
    noise. What moves is *reproducibility*: the volume bias of the plain runs was +4.5%,
    +12.2% and -11.6% across seeds, and with averaging it was -0.1%, +4.2% and +2.1%. A
    spread 5.7 times tighter, and paired per seed the distance from unbiased improved on
    all three, by 7.34% at 4.9 standard errors.

    Which is what averaging is supposed to do and not what one looks for in a table of
    scores: a point in the middle of a flat region is the same point whichever corner the
    last epoch wandered into. Read it as "the same model twice" rather than "a better
    model".

    No `monitor`: like `cosine_annealing` this is a function of how far through the run you
    are, not of how the run is going.
    """

    name: Literal["swa"] = Field("swa", description=_CALLBACK_NAME)
    start: float = Field(
        0.75,
        gt=0,
        le=1,
        description=(
            "The fraction of the run after which averaging begins - 0.75 folds the last "
            "quarter. A fraction rather than an epoch so that it survives a change to "
            "`epochs`, and the final epoch is always folded, so 1.0 averages one set of "
            "weights rather than none."
        ),
    )
    start_epoch: int | None = Field(
        None,
        ge=1,
        description=(
            "Do not act before this epoch, 1-based. Mutually exclusive with `start`. This is the one to use when the warm-up is a property of the optimisation rather than of the run's length - a schedule's warm-up is fifteen epochs whether the run is 60 or 600. For `swa` that is the epoch averaging begins at, and it is the "
            "same concept `early_stopping` names with these two words."
        ),
    )
    learning_rate: float | None = Field(
        None,
        gt=0,
        description=(
            "Hold the rate at this value once averaging begins. Unset leaves whatever the "
            "run was doing.\n\n"
            "Worth setting, and the reason is the whole mechanism: averaging is only worth "
            "something while the weights are still moving. A rate that has decayed towards "
            "zero produces a set of nearly identical snapshots, and their average is the "
            "last one."
        ),
    )

    @model_validator(mode="after")
    def one_way_of_saying_when_to_begin(self):
        """`start` or `start_epoch`, never both.

        `start` carries a default, so "was it set" cannot be read off its value - 0.75 is
        both the default and a perfectly good thing to write. `model_fields_set` is what
        pydantic knows about the configuration as it was actually written, which is the
        only thing that can tell the two apart.
        """
        if "start" in self.model_fields_set and self.start_epoch is not None:
            raise ValueError(
                "swa takes `start` or `start_epoch`, not both: `start` is a fraction of "
                "the run and `start_epoch` is an epoch number."
            )
        return self


class StopAfter(_Callback):
    """End the run on the clock, whatever the epochs say.

    An epoch cap is the wrong instrument for a wall-clock budget: how long an epoch takes
    is a property of the data and the shape, and the cases where this matters are exactly
    the ones where that was not what anybody expected. A card shared between queued
    measurements is the usual reason - a run that overshoots does not cost itself, it costs
    whatever was waiting behind it.

    **Stopping on the clock is not stopping on evidence**, so the weights at that moment
    are whichever epoch happened to be last. Pair it with `early_stopping`'s
    `restore_best`, or with `model_checkpoint`, if what the run produced matters.
    """

    name: Literal["stop_after"] = Field("stop_after", description=_CALLBACK_NAME)
    hours: float = Field(
        gt=0,
        description=(
            "The budget, in hours. Checked at the end of each epoch, so the real limit is "
            "this plus one epoch - said here rather than discovered, because on a long arm "
            "that remainder is twenty minutes. The run's `stop_reason` names the clock, so "
            "a run that ran out of time does not read as a run that converged."
        ),
    )


class TerminateOnCollapse(_Callback):
    """End the run when the model stops predicting any foreground at all.

    `terminate_on_nan`'s quiet sibling. A model that collapses to background is not
    obviously broken from the numbers: with `include_background` the Dice is a mean over
    class channels and the background channel is nearly perfect, so a collapsed model can
    report around 0.5 while the loss falls. The run completes, the table looks plausible
    and the masks are empty.

    What it watches is `val_foreground`, which every epoch reports: the share of the
    prediction that is not background.
    """

    name: Literal["terminate_on_collapse"] = Field(
        "terminate_on_collapse", description=_CALLBACK_NAME
    )
    floor: float = Field(
        0.0001,
        ge=0,
        lt=1,
        description=(
            "Below this share of foreground the prediction counts as collapsed. A floor "
            "rather than exactly zero, because a model predicting 0.01% foreground on a "
            "target that is 14% foreground has collapsed in every sense that matters. The "
            "default is deliberately far below any real target: this is a guard against a "
            "failure, not a quality threshold, and a floor near a real foreground share "
            "would end runs that were merely bad."
        ),
    )
    patience: int = Field(
        3,
        ge=1,
        description=(
            "Epochs in a row below the floor before the run ends. An early epoch "
            "legitimately predicts almost nothing, which is why this is not 1."
        ),
    )
    monitor: str = Field(
        "val_foreground",
        description=(
            "Which foreground share to watch. `train_foreground` is the other one, and it "
            "answers a different question: a model collapsing on the training data has a "
            "different problem from one collapsing only on data it has not seen."
        ),
    )


class Ema(_Callback):
    """An exponential moving average of the weights, updated every optimiser step.

    `swa`'s sibling and the other answer to the same observation: the last epoch's corner
    is not the middle of the basin. `swa` averages the tail uniformly from a declared
    start; this weights recent weights more and needs no start at all, because an average
    with a decay forgets the beginning by itself.

    **Updated per step, which is what makes it the estimator the literature means.** A
    decay of 0.999 is 0.999 per *step*: over 60 epochs of 200 batches that is 12,000
    updates and a half-life of about 700 steps. Announced per epoch the same number would
    do almost nothing, and offering it that way would have been publishing a name whose
    documented values mean something else - which is why `0.8.0a37` added a fourth
    callback hook rather than approximating one.

    Buffers are averaged too, which is the difference from `swa`: there a pass over the
    training data recomputes batch-normalisation statistics at the end, because a uniform
    average of weights has no matching statistics. An exponential average that also
    averages the buffers carries its own, so it needs no second pass over the data.

    One averager per run: `ema` and `swa` together are refused rather than silently
    applied in the order the list happened to have them.
    """

    name: Literal["ema"] = Field("ema", description=_CALLBACK_NAME)
    decay: float = Field(
        0.999,
        gt=0,
        lt=1,
        description=(
            "How much of the average survives each step. 0.999 is the usual value and "
            "means a half-life of about 700 steps - roughly three and a half epochs at "
            "200 batches each. **Per step, not per epoch**, so the published values "
            "transfer: a run with few steps wants a smaller number, because an average "
            "that forgets more slowly than the run is long never leaves its starting "
            "point. The model is replaced by the average when training ends."
        ),
    )


class CsvLogger(_Callback):
    """The history, written as it arrives.

    The same numbers `fit()` returns, one row per epoch, on disk the moment the epoch ends -
    so a run killed at epoch 40 of 60 still leaves behind what it learned. Until 0.8.0a23 this
    sentence named a function from the R package instead, which is a dead end for the reader
    the configuration reference is written for.
    Cheap enough that the only reason not to add it is having nowhere to write.
    """

    name: Literal["csv_logger"] = Field("csv_logger", description=_CALLBACK_NAME)
    path: str = Field(
        description=(
            "Where to append one row per epoch. The same numbers `fit()` returns, written "
            "as they arrive - so a run that is interrupted still leaves its history behind."
        ),
    )


class TerminateOnNaN(_Callback):
    """End the run the moment the loss stops being a number.

    No fields: there is nothing to decide. What it saves is the rest of a run that cannot
    recover, and NaN here has causes worth knowing rather than fearing - a learning rate
    far too high, a metric with `smooth: 0` meeting a class that is absent from an image,
    or a skeleton-based score on an empty mask.
    """

    name: Literal["terminate_on_nan"] = Field("terminate_on_nan", description=_CALLBACK_NAME)


CallbackSpec = Annotated[
    EarlyStopping
    | ModelCheckpoint
    | ReduceLrOnPlateau
    | CosineAnnealing
    | Ema
    | LinearAnnealing
    | StepDecay
    | StopAfter
    | Swa
    | TerminateOnCollapse
    | CsvLogger
    | TerminateOnNaN,
    Field(discriminator="name"),
]

# --------------------------------------------------------------------- augmentation
# Deliberately not a discriminated union. albumentations ships 131 transforms with dozens
# of parameters each, and mirroring them here would be a second, always slightly stale
# copy of someone else's API. A hardcoded list proved that point immediately: written
# against albumentations 1.1 it already named `Flip`, which 2.x removed.
#
# So the names are read from the installed albumentations, and the parameters are checked
# by albumentations itself when the pipeline is built. The import is lazy and cached, so
# a spec with no augmentation never pays for it.
#
# PLAN.md rule 4 said albumentations was "today's 2D implementation" and 3D would need a
# different backend. That turns out to be too pessimistic: albumentations 2.x carries 3D
# transforms (CenterCrop3D, CoarseDropout3D, CubicSymmetry), so the same backend may well
# serve both ranks. The interface stays either way.


@lru_cache(maxsize=2)
def available_transforms(rank: int = 2) -> frozenset[str]:
    """Transform names the installed albumentations actually offers.

    With `rank=3`, only those that can transform a volume. That list has to be found by trying
    rather than read from anywhere: albumentations supports volumes unevenly, and a transform
    that cannot raises from inside itself - `GaussNoise` comes back as `KeyError: 'images'`.
    albumentations 2.0.8 offers 118 transforms, and 91 of them took a volume where this was
    measured.

    **That second number is not portable and the first one is.** 118 is a property of the
    installed albumentations; the volume count is the outcome of running each transform against
    a probe, and some of them fail it for reasons that have nothing to do with volumes. What the
    27 refusals actually are, counted on albumentations 2.0.8 rather than reasoned about:

        10   no volume support, which is the answer being asked for - `KeyError: 'images'`
         9   require three channels, where the probe has one
         8   require a target the probe does not supply: seven ask for metadata of their own
             (`FDA`, `HistogramMatching`, `Mosaic`, `OverlayElements`,
             `PixelDistributionAdaptation`, `RandomCropNearBBox`, `MaskDropout`) and
             `BBoxSafeRandomCrop` asks for bounding boxes

    So **17 of the 27 are the probe's limits rather than albumentations'**, and only 10 are a
    real no. The size category used to be a fourth entry in that table and is gone: a refusal is
    now retried at a 1024x1024 probe, which is what `Crop`'s default crop box needs, and the
    four it was hiding - `Crop`, `FrequencyMasking`, `TimeMasking`, `Superpixels` - are listed.
    The three-channel nine are deliberately still refused; the comment above the probe says why.

    **The old count moved between runs, and the reason was the probe, not the platform.** One
    CI matrix answered 87 on four Linux builds, both macOS builds and Windows with Python 3.10
    - 2026-10-11, on a matrix that still had a 3.10 cell -
    and 88 on Windows with Python 3.13, which was recorded as that combination with no mechanism
    claimed. There is no platform axis: `Superpixels` samples `n_segments`, and at 8x8 whether
    the segmentation divides by zero depends on the draw - 4 of 60 attempts succeed. Eight
    consecutive listings in one process on one machine gave 87, 87, 87, 87, 87, 87, 88, 87.

    Escalating removes that, because every refusal is retried where all four size-dependent
    transforms pass deterministically - `Superpixels` 40 times out of 40 at 1024x1024. Eight
    listings gave 91 eight times. The number is still not pinned: it is a property of the
    installed albumentations, and "stable on one machine" is not "portable". The authoritative
    check is the one made when the pipeline is built, against the user's own shape and
    parameters.

    Empty when albumentations is missing, in which case name checking is skipped and the
    backend reports the problem later - better than refusing a spec the user cannot fix.

    Args:
        rank: 2 for images, 3 for volumes.

    Returns:
        The names, as albumentations spells them, and only things a spec may actually
        name. Membership is decided by type - `BasicTransform` - so the composition
        classes (`Compose`, `OneOf`, `Sequential` and five more) and the two parameter
        dataclasses (`BboxParams`, `KeypointParams`) are absent. They were listed
        until 0.7.0a2, when the filter went by the shape of the name, and none of
        them was ever usable: an `AugmentationStep` is a flat name and a dict of
        parameters with no nesting, so naming one passed the check here and failed
        when the pipeline was built.

    >>> names = available_transforms()
    >>> "HorizontalFlip" in names, "GaussNoise" in names
    (True, True)
    >>> any(name in names for name in ("Compose", "OneOf", "BboxParams"))
    False

    At rank 3 the list is shorter, and it is found by trying rather than read from
    anywhere: support for volumes is uneven and a transform that cannot take one
    raises from inside the library.

    >>> volumes = available_transforms(rank=3)
    >>> len(volumes) < len(names), "GaussNoise" in volumes
    (True, False)
    """
    try:
        import albumentations
    except ImportError:  # pragma: no cover - exercised only without the optional backend
        return frozenset()

    names = frozenset(
        name
        for name in dir(albumentations)
        if _is_transform(albumentations, getattr(albumentations, name, None))
    )
    if rank != 3:
        return names
    return frozenset(name for name in names if _transforms_volumes(albumentations, name))


def _is_transform(albumentations, member: object) -> bool:
    """Whether this member of the albumentations namespace is a transform a spec may name.

    Type answers most of it: a transform is a `BasicTransform`, which the composition classes
    - `Compose`, `OneOf`, `Sequential` and the rest - are not, being `BaseCompose`, and which
    `BboxParams` and `KeypointParams` are not, being parameter dataclasses. None of the ten is
    usable here anyway, because an `AugmentationStep` is a flat name and a dict of parameters
    with no nesting, so naming one passed the check and then failed when the pipeline was
    built.

    The four interface classes are named rather than detected, because **nothing in their type
    says they are interfaces** - measured, not assumed. `DualTransform()`, `ImageOnlyTransform()`
    and `Transform3D()` all instantiate without complaint; forty genuine transforms inherit
    `apply` instead of defining it, so "has no own `apply`" is no use either; and "is a
    superclass of another public transform" catches `Affine`, `Blur`, `HorizontalFlip`, `NoOp`,
    `Pad` and `D4`, which are all usable. So there is no rule to derive, and a list of four
    identities is the honest form - with a test pinning the resulting count, so that a base
    added upstream shows up as a failure rather than as a silent extra name.

    `NoOp` is why the module cannot decide it either: it is a real transform living in
    `albumentations.core.transforms_interface` beside the four.
    """
    if not isinstance(member, type) or not issubclass(member, albumentations.BasicTransform):
        return False
    interfaces = (
        albumentations.BasicTransform,
        albumentations.DualTransform,
        albumentations.ImageOnlyTransform,
        getattr(albumentations, "Transform3D", None),
    )
    return member not in [cls for cls in interfaces if cls is not None]


# The probe `_transforms_volumes` offers each transform. Two sizes rather than one, because
# a single size cannot be both cheap and large enough: 8x8 is where the whole listing is
# nearly free, and 1024x1024 is where every size-dependent default fits. So the small one is
# tried first and only a refusal is retried large.
#
# The cost, as a ratio because seconds are not portable and this is a cached call made once:
# on one Linux machine the escalating listing took **8x** the 8x8-only listing, and offering
# everything the large probe would have taken **250x**. Most of the 8x is `Superpixels`, whose
# work scales with the pixel count; the ratio is the part worth trusting, not the factors.
#
# The depth is 4 and is not escalated: measured, not assumed. Nothing in the rejected set
# changes its answer at depth 16, so there is no second threshold hiding behind the first.
#
# The channel count stays at 1 on purpose, and that is a different decision rather than an
# unfinished one. Nine transforms refuse a single channel, and probing with three would list
# `ChromaticAberration` and its eight relatives as available to someone whose CT has one
# channel. A listing cannot know the channel count; only the pipeline-build check can, and it
# already probes at the model's own input size.
_PROBE_DEPTH = 4
_PROBE_SIDES = (8, 1024)


def _transforms_volumes(albumentations, name: str) -> bool:
    """Whether this transform can be ruled out for volumes. Tried, because nothing declares it.

    A transform that needs arguments - `CenterCrop3D` wants a size, and it is one of the
    3D-native ones - cannot be probed with defaults, and is kept rather than dropped. "I could
    not check" is not "it does not work", and dropping those would have hidden exactly the
    transforms written for volumes. The authoritative check happens when the pipeline is built,
    against the parameters the user actually gave.

    **A refusal is retried at the larger probe before it is believed**, for the same reason: a
    transform whose defaults do not fit inside an 8x8 image has told us about the probe and
    nothing about volumes. Four said so - `Crop`, whose default crop box is 1024x1024, and
    `FrequencyMasking`, `TimeMasking` and `Superpixels`, which need 32, 64 and 16 pixels
    respectively. Escalation can only add names, never remove them, since a transform that
    passed at 8x8 is never asked again.

    Warnings are silenced: albumentations advises about aliases and slow implementations, and a
    listing that emits twenty warnings is a listing nobody reads.
    """
    import warnings

    cls = getattr(albumentations, name, None)
    if cls is None:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            transform = cls(p=1)
        except Exception:  # noqa: BLE001 - needs arguments, so it cannot be ruled out here
            return True
        return any(_accepts_volume(albumentations, transform, side) for side in _PROBE_SIDES)


def _accepts_volume(albumentations, transform, side: int) -> bool:
    """Whether this transform survives one pass over a `side` x `side` probe volume.

    Called inside `catch_warnings`, so it does not silence anything itself.
    """
    import numpy as np

    probe = np.zeros((_PROBE_DEPTH, side, side, 1), dtype=np.float32)
    mask = np.zeros((_PROBE_DEPTH, side, side), dtype=np.uint8)
    try:
        albumentations.Compose([transform])(volume=probe, mask3d=mask)
    except Exception:  # noqa: BLE001 - any refusal is a refusal; the reason is not parsed
        return False
    return True


class AugmentationStep(SpecModel):
    name: str = Field(
        description=(
            "An albumentations transform, by its own class name - `HorizontalFlip`, "
            "`RandomBrightnessContrast`. Checked against what the installed albumentations "
            "offers, and `available_transforms()` lists them; at rank 3 the list is shorter, "
            "because support for volumes is uneven."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments passed to that transform, exactly as albumentations names them - so "
            "`p` is the probability of applying it, defaulting to the transform's own. Not "
            "validated here: they are albumentations' to accept, and each step is probed at "
            "the model's input size when the pipeline is built, so a bad argument is named "
            "there rather than guessed at here."
        ),
    )

    @field_validator("name")
    @classmethod
    def known_transform(cls, value: str) -> str:
        known = available_transforms()
        if not known or value in known:
            return value
        close = get_close_matches(value, known, n=3, cutoff=0.6)
        if not close:
            # get_close_matches is case-sensitive; a pure casing slip deserves a hit too.
            lowered = value.lower()
            close = [name for name in known if name.lower() == lowered]
        suggestion = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"unknown augmentation '{value}'.{suggestion}")
        return value
