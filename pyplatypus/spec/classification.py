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

**One label per image, and multi-label is a different feature rather than a setting.** A
chest film with four findings is not this - it is `classes_from`'s shape one task across,
an independent question per class rather than one of a set of alternatives, with sigmoid
channels and a loss that has to support them. `supports_multilabel` already exists on every
loss for exactly that distinction, so the day it lands it reads the flag that is there. Said
here because a reader who wants it should learn it from this file rather than from a
refusal.

**What this is not** is in `pyplatypus#206`'s own words: not a general-purpose image
classifier. It exists because `crops()` cuts objects out and hands them to "a classifier it
does not contain", and because explainability needs a per-image answer to explain.

Nothing in this file is reachable from a specification yet: `task: classification` lands
last, after the head and the metrics, which is the order `0.8.0a20` built the instance task
in and the order `tests/spec/test_task_coverage.py` exists to make safe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyplatypus.spec.components import CceLoss, LossSpec
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
            "leaves nothing, so the same setting quietly changes how much data there is."
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
        """The CSV column holding the label. Derived, like every other task's."""
        return "label"

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

    @property
    def architecture(self) -> str:
        """What kind of network this is, derived from `encoder` rather than declared.

        A string rather than an enum member, which is the one place this task reads
        differently from its siblings: `SegmentationModel.architecture` and
        `DetectionModel.architecture` are fields holding enums and carry `.value`. There is
        no enum to be a member of here - the set of backbones is timm's, and ours is the
        absence of one - so anything printing this prints the string.
        """
        return self.encoder or "built_in"

    @property
    def monitorable(self) -> set[str]:
        """The loss, and not `foreground`.

        `ModelSpec.monitorable` offers `train_foreground` and `val_foreground` because a
        segmentation epoch reports the share of the prediction that is not background, which
        is what `terminate_on_collapse` watches. Classification has no background class and
        no such number, so a callback watching it is refused by name here rather than
        failing at the end of the first epoch with a `KeyError` out of `state.logs` - which
        is exactly what `0.8.0a27` cost, one metric across.

        The metrics are not here yet, deliberately: accuracy is the wrong default on medical
        data - a classifier answering 'no finding' where 3% have one scores 0.97 - and
        choosing what *is* the default is its own slice. Offering segmentation's overlap
        metrics in the meantime would be offering a choice that does not exist.
        """
        return {"train_loss", "val_loss"}

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
                f"Remove them to use the backbone, or remove `encoder` to build ours."
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
