"""One label for a whole image: the classification half of the specification.

The two questions this task adds are the same two every task here adds - what the classes
are called, and where the labelling is written - and both are answered differently from
anywhere else in this package. `labels` and `colormap` describe a mask, `classes` in a
detection spec describes a file of boxes; here the label is one name per image and it
arrives either as the directory the image sits in or as a column in a CSV.

**Which of those two it is follows from `mode` rather than being said twice.** A classifier
in `nested_dirs` mode reads the image's parent directory as its class - the layout every
image dataset ships in - and in `config_file` mode reads the `label` column. That is a
second reading of `nested_dirs`, and it is the same reading the other tasks give it: the
directory tree carries the labelling, and what the tree carries is the task's own business.
What keeps the two from colliding is `subdirs`, which is refused here, so a per-sample
layout cannot be written for a classification spec at all.

**A label may also be a SET, and it is a naming rather than a setting.** A chest film with
four findings is `classes_from`'s shape one task across - an independent question per class
rather than one of a set of alternatives - and `labels_from` is how it is said: one CSV
column per class, in class order. It reads the `supports_multilabel` flag that was already
on every loss and metric for exactly this distinction rather than keeping a list, which is
what that paragraph promised when it said *the day it lands*. `pyplatypus#307`.

The reason it cannot be a switch is the reason the mask side gives: **the target changes,
not only the activation.** A `label` column structurally cannot say that a film has two
findings, so `activation: sigmoid` over an exclusive target would train, raise nothing, and
report plausible numbers meaning what softmax meant.

**What this is not** is in `pyplatypus#206`'s own words: not a general-purpose image
classifier. It exists because `crops()` cuts objects out and hands them to "a classifier it
does not contain", and because explainability needs a per-image answer to explain.

Reachable from a specification since `0.8.0a49`, and the tag landed **last** - after the
head, the metrics, the reader and the trainer - which is the order `0.8.0a20` built the
instance task in and the order `tests/spec/test_task_coverage.py` exists to make safe. That
test fired on the first edit of a49, when `Task` gained a member and `AnySpec` had not, which
is the window it was written for; the paragraph that stood here said *nothing in this file is
reachable yet* and was true until 21:43Z on 2026-10-27.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyplatypus.spec.common import DataMode
from pyplatypus.spec.components import (
    AucMetric,
    CceLoss,
    ClassificationMetricSpec,
    LossSpec,
)
from pyplatypus.spec.data import DataSpec
from pyplatypus.spec.models import EncoderModel


def _admissible_losses() -> str:
    """The losses a label can be scored with, read off the flags rather than listed.

    Imported here rather than at module level because `catalogue` imports this module's
    siblings; the cost is one import on a path that only runs while a refusal is being
    written, which is not a path anybody waits on.
    """
    from pyplatypus.spec.catalogue import _members

    return ", ".join(
        member.model_fields["name"].default
        for member in _members(LossSpec)
        if not member.needs_extent and not member.continuous
    )


class ClassificationData(DataSpec):
    """Where the images are, and what their labels are called.

    `classes` is this task's answer to the colormap and it is stated for the same reason
    detection states its own: **the position of a name in this list is the class index the
    model learns.** Read off the directories instead, the order would come out alphabetical
    - reproducible, and unrelated to anything - so a model trained where index 0 meant
    `benign` and used where it means `malignant` reports plausible probabilities and nothing
    downstream can tell.

    Two classes at a minimum, where detection allows one. A detector with one class answers
    *where are the cells*, which is a question; a classifier with one class answers *is this
    a cell* with the only answer it has, which is not. The refusal is cheap and the mistake
    it catches - a `classes` list that lost its second entry in an edit - is not visible in
    any number a run produces.
    """

    subdirs: tuple[str, str] | None = Field(
        None,
        description=(
            "Refused here, and it is the field that keeps `nested_dirs` from meaning two "
            "things at once. For segmentation and detection a sample is a directory with "
            "the image in one subdirectory and what labels it in the other; a classification "
            "sample is one image whose label is the directory it sits in, so there is no "
            "second subdirectory to name. Leave it unset: in `nested_dirs` mode the classes "
            "are the directories under `train_path`, and in `config_file` mode the `label` "
            "column holds the name."
        ),
    )

    classes: list[str] = Field(
        min_length=2,
        description=(
            "Class names in class order. Position is the class index, and in `nested_dirs` "
            "mode these are the directory names under `train_path`. No background entry: an "
            "image is one of these, and a picture of nothing in particular is a class you "
            "have to name."
        ),
    )
    strict_labels: bool = Field(
        True,
        description=(
            "Refuse an image whose label is not in `classes`. False skips it instead, which "
            "is how you train on two of a dataset's three classes without moving any files. "
            "True by default, and it matters more here than it does for a box: a skipped "
            "object leaves the rest of its image in the training set, while a skipped image "
            "leaves nothing, so the same setting quietly changes how much data there is.\n\n"
            "Has nothing to choose between when `labels_from` is given - the columns *are* "
            "the classes, so no cell can hold a class nobody named - and `false` is refused "
            "there rather than ignored."
        ),
    )
    labels_from: list[str] | None = Field(
        None,
        min_length=2,
        description=(
            "For a label that is a **set** - one CSV column per class, in class order, each "
            "cell `0` or `1`. A chest film with four findings is four independent questions "
            "rather than one of four alternatives, and this is the only way of saying so.\n\n"
            "It is a naming rather than a switch, for the reason `regions` and "
            "`classes_from` are on the mask side: **the target changes, not only the "
            "activation**. A `label` column structurally cannot say that a film has two "
            "findings, so a setting that turned the softmax into a sigmoid would train "
            "sigmoid channels against an exclusive target, raise nothing, and report "
            "plausible numbers meaning exactly what softmax meant. The naming carries the "
            "claim, and everything downstream reads `multilabel`.\n\n"
            "The position in this list is the class index, paired with `classes` - so the "
            "headers may be called whatever the dataset calls them and the model still "
            "learns the classes in the order you named. Stated rather than read off the "
            "header row for `channels_from`'s reason: sorting is reproducible and "
            "clinically meaningless.\n\n"
            "`config_file` mode only, and that is a refusal rather than a gap: an image "
            "cannot sit in two directories without being two files, so `nested_dirs` cannot "
            "express a set of labels at all.\n\n"
            "A cell that is not `0`, `1`, `0.0` or `1.0` is refused by name, and the case "
            "that is written for is `-1`. The datasets shaped this way write `-1` for "
            "*uncertain* and leave the cell empty for *not mentioned*; both have a "
            "defensible mapping to 0 or to 1, which is exactly why this reader must not "
            "pick one for you. An image with none of the findings is an ordinary labelled "
            "row, written as a line of zeros.\n\n"
            "**There is no threshold field.** Each class decides for itself at `logits > 0`, "
            "which is a probability of 0.5 exactly, and a threshold is a number about your "
            "data rather than one this format should invent. And `accuracy` is refused "
            "outright here rather than redefined: measured on 4000 images with findings at "
            "10%, 5%, 3% and 2%, a model answering no to everything scores 0.8137 on exact "
            "match and 0.9507 per class, **above a model holding half the available "
            "signal** - so neither reading of it can be what a `monitor` watches. Use "
            "`balanced_accuracy`, `f1` or `auc`, all of which put that answer below every "
            "model with any signal at all."
        ),
    )

    @model_validator(mode="after")
    def the_label_is_the_directory_or_a_column(self):
        """`subdirs` is the one field of `DataSpec` this task cannot answer.

        Refused rather than ignored, for the reason every refusal in this package is: a
        setting that does nothing leaves somebody believing they have described their
        layout. The message carries the layout instead, because that is what the person
        writing `subdirs` was trying to say.
        """
        if self.subdirs is not None:
            raise ValueError(
                "`subdirs` names the two subdirectories of a sample directory, and a "
                "classification sample is a single image. In `nested_dirs` mode the "
                "directories under the path are the classes - one per class, images "
                "inside - and in `config_file` mode the CSV's `label` column holds the "
                "name. Remove `subdirs`."
            )
        return self

    @model_validator(mode="after")
    def a_set_of_labels_is_read_from_columns_and_not_from_directories(self):
        """`labels_from` needs the layout that can hold several answers for one image.

        The directory tree carries exactly one answer per file, so this is not a gap in the
        reader - there is nothing for it to read. The message names the layout, the way
        `subdirs`' refusal does, because that is what the person who wrote the field was
        trying to describe.
        """
        if self.labels_from is None or self.mode is not DataMode.NESTED_DIRS:
            return self
        raise ValueError(
            "`labels_from` reads one column per class, which is `config_file` mode. In "
            "`nested_dirs` mode the directory an image sits in *is* its label, and an image "
            "sits in one directory - so a set of labels cannot be written that way at all. "
            "Move to a CSV with one column per class, or drop `labels_from` and name one "
            "class per directory."
        )

    @model_validator(mode="after")
    def one_column_per_class_in_class_order(self):
        """As many columns as classes, because the pairing is by position.

        A shorter list is not a subset - it is a different pairing, and the model would
        learn classes whose names are somebody else's. Checked here rather than in the
        reader so that it is refused while a configuration is being read rather than when a
        dataset is walked.
        """
        if self.labels_from is None or len(self.labels_from) == len(self.classes):
            return self
        raise ValueError(
            f"`labels_from` names {len(self.labels_from)} column(s) and `classes` names "
            f"{len(self.classes)}; there is one column per class and the position in both "
            f"lists is the class index, so a shorter list is a different pairing rather "
            f"than a subset."
        )

    @model_validator(mode="after")
    def nothing_to_be_strict_about_when_the_columns_are_the_classes(self):
        """`strict_labels: false` asks for a state `labels_from` cannot produce.

        It chooses what to do with an image whose label is not a named class, and with one
        column per class there is no cell that could hold one. `true` written is a redundant
        statement of what happens anyway and is left alone; only `false` asks for something
        this naming cannot give.

        **`model_fields_set` is not consulted and that is measured rather than assumed.** The
        mask side's `include_background: false` has to ask, because `cldice` defaults that
        field to False and a default is not a request. Here the default is `True` for every
        model, so `strict_labels is False` **is** the evidence that somebody wrote it - the
        check was here for an hour and the mutation that deletes it was caught by nothing,
        because it is unreachable.
        """
        if self.labels_from is None or self.strict_labels:
            return self
        raise ValueError(
            "`strict_labels: false` skips an image whose label is not one of `classes`, and "
            "with `labels_from` there is no such image: the columns are the classes, paired "
            "by position, so a cell can only say yes or no to a class you named. Leave the "
            "setting out. A cell that is unreadable - `-1`, or empty - is refused by name "
            "rather than skipped, because an unreadable cell is not a missing label."
        )

    @model_validator(mode="after")
    def class_names_are_distinct(self):
        """Two classes sharing a name cannot be told apart, and in `nested_dirs` mode they
        cannot even be two directories."""
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(
                "class names must be distinct - two classes sharing a name are one class "
                "with the wrong index, and a directory name is unique anyway"
            )
        return self

    @property
    def label_column(self) -> str:
        """The CSV column holding the label. Derived, like every other task's.

        Not read at all when `labels_from` is given: the classes have a column each then,
        and a `label` column in the same file is one of the many columns the reader ignores.
        """
        return "label"

    @property
    def multilabel(self) -> bool:
        """Whether one image may carry several of these classes at once.

        The single switch the rest of the package reads - the loss's binary form, a sigmoid
        per class instead of a softmax across them, a two-by-two table per class instead of
        one confusion matrix, and a threshold instead of an `argmax`. `DataSpec.multilabel`
        is the same property one task over and is derived the same way: **from the naming**,
        because the naming is the only thing that can carry the claim.

        One naming answers yes here where two do on the mask side, and the asymmetry is the
        layout rather than a decision: `regions` derives overlap from one label map's values
        and `classes_from` reads it from separate files, while an image's own directory
        cannot say two things at once.
        """
        return self.labels_from is not None

    @property
    def n_class(self) -> int:
        """How many classes there are. `classes` decides; nothing else."""
        return len(self.classes)


class ClassificationModel(EncoderModel):
    """A contracting path, pooled, with one dense layer on top.

    **There is no `architecture` field, because for this task the architecture *is* the
    contracting path.** A segmentation model chooses between four decoders that read the
    same features, and that choice has a name; a classifier has nothing after the pooling
    but a dense layer, so `encoder` is the whole answer and a second field naming the
    network would be two places for one. `architecture` is a property here rather than a
    field for that reason, and `weights_fingerprint` reads it the same way it reads the
    others'.

    `blocks`, `filters` and `block_width` are declared here rather than inherited: the
    contracting path is one mechanism and `EncoderModel` holds every knob whose prose is
    the same for both tasks, but these three say things about segmentation that are false
    here - a size that must divide by 2^blocks, and a published weights file that fills
    them in - and a description that is true of one task and false of the other is worse
    than two descriptions.
    """

    blocks: int = Field(
        4,
        ge=1,
        le=8,
        description=(
            "How many times the resolution is halved before the features are pooled, so how "
            "much context the answer is made from. Unlike a segmentation model nothing has "
            "to divide by 2^blocks - there is no decoder to line up with - but every side of "
            "`input_shape` must still be at least 2^blocks, or the deepest level has nothing "
            "left to pool. Ignored, and refused, when `encoder` names a backbone: the "
            "backbone's own depth is then the only one there is."
        ),
    )
    filters: int = Field(
        16,
        ge=1,
        description=(
            "Filters in the first block, doubled at every level, so this and `blocks` set "
            "how wide the pooled feature vector is: filters * 2^blocks. Ignored, and "
            "refused, when `encoder` names a backbone."
        ),
    )
    block_width: int = Field(
        2,
        ge=1,
        le=4,
        description=(
            "Convolutions per block, at every level. Raising it adds depth without adding "
            "levels, which is the lever to reach for when the objects are small enough that "
            "halving the resolution again would lose them - a crop of one cell is exactly "
            "that case. Ignored, and refused, when `encoder` names a backbone."
        ),
    )

    pooling: Literal["avg", "max"] = Field(
        "avg",
        description=(
            "How the feature map becomes one vector per image. `avg` is what every "
            "classification head since the networks that replaced the flattened dense layer "
            "uses, and it averages evidence over the whole frame; `max` takes the strongest "
            "response anywhere, which is the one to reach for when the thing being looked "
            "for is small and somewhere in the picture rather than the picture itself."
        ),
    )
    head_dropout: float = Field(
        0.0,
        ge=0,
        lt=1,
        description=(
            "Drop this fraction of the pooled vector before the dense layer. Separate from "
            "`dropout`, which sits between convolutions and reaches nothing at all when the "
            "contracting path is a pretrained backbone with its own regularisation - this "
            "is then the only one left, which is why it is here rather than folded into the "
            "other."
        ),
    )

    loss: LossSpec = Field(
        default_factory=CceLoss,
        description=(
            "The objective trained against. Fewer are admissible here than anywhere else in "
            "this package, and the two that are - cross-entropy and its focal form - are the "
            "ones that score each answer on its own rather than scoring an overlap. The "
            "refusal names them rather than this description keeping a list."
        ),
    )

    metrics: list[ClassificationMetricSpec] = Field(
        default_factory=lambda: [AucMetric()],
        description=(
            "What to report beside the loss. **`auc` rather than `accuracy`, and that is a "
            "measurement rather than a preference**: on 40000 images at a prevalence of 3%, "
            "with the model's own probabilities as the truth so that only its "
            "discrimination varies, accuracy spans 0.0100 where the AUC spans 0.4631 and is "
            "not even monotone - the model with no signal scores above two models that have "
            "one. Balanced accuracy and F1 do not rescue it either: a calibrated model at "
            "that prevalence never crosses the threshold, so every metric read off the hard "
            "prediction sees the same all-negative answer. Being threshold-free is what "
            "separates them. A callback watching `val_<metric>` is refused when nothing "
            "here will produce it."
        ),
    )

    @property
    def architecture(self) -> str:
        """What kind of network this is, derived from `encoder` rather than declared.

        A string rather than an enum member, which is the one place this task reads
        differently from its siblings: `SegmentationModel.architecture` and
        `DetectionModel.architecture` are fields holding enums and carry `.value`. There is
        no enum to be a member of here - the set of backbones is timm's, and ours is the
        absence of one - so anything printing this prints the string.

        **`plain_cnn` names the shape rather than the owner**, which is what every other
        value this field can hold does: `u_net`, `linknet` and `res_u_net` on the
        segmentation side, and timm's whole catalogue on this one. It was `built_in` first,
        and that said *not somebody else's* where every neighbour says *what* - and it would
        have had nothing left to say the day a second one of ours arrives, while `plain_cnn`
        beside a `residual_cnn` is two shapes. It is also the line `_encoder_for` already
        draws: ours is the plain convolution stack rather than the residual one, because
        `encoder: resnet34` is the residual network in one word. Checked against timm's own
        catalogue, since this field's values mix with theirs - nothing there is called it -
        and checked by hand, which nothing holds.

        **The rename is free exactly until it is not.** `weights_fingerprint` writes this
        string into a published file's sidecar and compares it on load, so after the first
        published classification weights a rename refuses every one of them by field name.
        """
        return self.encoder or "plain_cnn"

    @property
    def monitorable(self) -> set[str]:
        """The loss, and not `foreground`.

        `ModelSpec.monitorable` offers `train_foreground` and `val_foreground` because a
        segmentation epoch reports the share of the prediction that is not background, which
        is what `terminate_on_collapse` watches. Classification has no background class and
        no such number, so a callback watching it is refused by name here rather than
        failing at the end of the first epoch with a `KeyError` out of `state.logs` - which
        is exactly what `0.8.0a27` cost, one metric across.

        **Every metric asked for, and no `_worst`.** A segmentation epoch also offers
        `val_<metric>_worst`, the figure `summarise_cases` tells every reader to look at,
        because a case there is an image with hundreds of thousands of voxels and its own
        score. One image's accuracy is 0 or 1, so the worst case of a split is 0 for any
        model that gets a single image wrong - a number that carries no information and
        would read as one that does. The question it is asking - *which cases is this bad
        at* - is a per-class figure here, which is what `balanced_accuracy` and `f1` are.

        Nor is there a `foreground`: `ModelSpec` offers it because a segmentation epoch
        reports the share of the prediction that is not background, which is what
        `terminate_on_collapse` watches. Classification has no background class and no such
        number, so a callback watching it is refused by name here rather than failing at the
        end of the first epoch with a `KeyError` out of `state.logs` - which is exactly what
        `0.8.0a27` cost, one metric across.
        """
        return (
            {"train_loss", "val_loss"}
            | {f"train_{metric.name}" for metric in self.metrics}
            | {f"val_{metric.name}" for metric in self.metrics}
        )

    def weights_fingerprint(self) -> dict:
        """`blocks` and `filters` only when the contracting path is ours.

        They are what identifies one of our encoders, the way they identify a U-shaped
        model. A named backbone builds none of it, so including them would refuse a file
        over two fields that shaped nothing - and `architecture`, which is the backbone's
        name, already separates `resnet34` from `efficientnet_b0`.
        """
        fingerprint = super().weights_fingerprint()
        if self.encoder is None:
            fingerprint |= {"blocks": self.blocks, "filters": self.filters}
        return fingerprint

    @model_validator(mode="after")
    def the_contracting_path_is_ours_or_theirs(self):
        """Our encoder's knobs, written beside somebody else's backbone, reach nothing.

        A segmentation model with an `encoder` still uses `blocks` - it sizes the decoder -
        so there the two coexist. Here the backbone is the whole network below the pooling,
        and `blocks: 6` beside `encoder: resnet34` builds a resnet34 and says nothing. Read
        off `model_fields_set` rather than off the values, so a default that happens to
        differ is not mistaken for an instruction: the question is whether somebody wrote
        it.
        """
        if self.encoder is None:
            return self
        written = sorted(
            field
            for field in ("blocks", "filters", "block_width")
            if field in self.model_fields_set
        )
        if written:
            joined = ", ".join("`" + field + "`" for field in written)
            raise ValueError(
                f"model '{self.name}' names `encoder: {self.encoder}` and {joined}. Those "
                f"describe the built-in contracting path, and a backbone replaces it "
                f"entirely - a classifier has nothing after the pooling for them to shape. "
                f"Remove them to use the backbone, or remove `encoder` to build ours - "
                f"which is the path `plain_cnn` names."
            )
        return self

    @model_validator(mode="after")
    def every_side_survives_the_pooling(self):
        """`blocks` pools `blocks` times, and a side of 1 cannot be halved again.

        Segmentation refuses the same arithmetic one step earlier, by requiring the size to
        divide exactly, because its decoder has to line up with what the encoder skipped.
        Nothing here is upsampled, so an odd size is fine and only disappearing is not: at
        `blocks: 5` a 30-pixel side goes 15, 7, 3, 1 and then has nothing left to pool,
        which is a shape error from inside the network rather than anything a reader of the
        specification would see.
        """
        if self.encoder is not None:
            return self
        smallest = 2**self.blocks
        bad = [size for size in self.input_shape if size < smallest]
        if bad:
            raise ValueError(
                f"with blocks={self.blocks} every spatial dimension must be at least "
                f"{smallest}; input_shape={tuple(self.input_shape)} is not. Nothing has to "
                f"divide exactly here - there is no decoder to line up with - but a side "
                f"smaller than that runs out of pixels before the last level."
            )
        return self

    @model_validator(mode="after")
    def the_label_needs_a_loss_over_classes(self):
        """A regression loss over a label, refused for the reason `0.8.0a21` refused it over
        a mask: `loss` is the objective on the class channels, and a label is a set of
        classes rather than a number. Read off the loss's own `continuous` flag, so a fourth
        regression loss is refused without this file being edited."""
        if not self.loss.continuous:
            return self
        raise ValueError(
            f"loss '{self.loss.name}' scores a continuous target, and the target here is "
            f"one of the alternatives `classes` names - a label, not a number. Use "
            f"{_admissible_losses()}."
        )

    @model_validator(mode="after")
    def an_overlap_needs_something_to_overlap(self):
        """An overlap loss on one label per image is smoothing constants and nothing else.

        `dice` and its family score the intersection of two regions against their union. A
        classification target has no region: the sample is one answer, so the 'overlap' is
        one probability against one 0 or 1, and what is left of the coefficient is the
        smoothing term that was there to stop it dividing by zero. It trains - that is the
        problem - and it reports a number in the usual range while being the wrong question.

        Read off `needs_extent`, which every loss declares, rather than a list here. The
        message names what is admissible from the same flag, so a loss added to the package
        joins the list by arriving.
        """
        if not self.loss.needs_extent:
            return self
        raise ValueError(
            f"loss '{self.loss.name}' is defined over a region - an overlap, a boundary, or "
            f"a ranking of the elements - and a classification target is one label for a "
            f"whole image, which has none. Use {_admissible_losses()}."
        )
