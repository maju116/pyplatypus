"""The whole experiment, and the single contract between R and Python.

Both `platypus_spec(...)` in R and a YAML file produce this object. Nothing downstream
can tell which one was used, which is the entire point.

`task` chooses between three of them. It is a discriminator rather than a setting because
it decides the *type* of `data` and of every entry in `models`: a detection spec holding
a `colormap`, or a segmentation spec holding `anchors`, is not a spec with a stray field
but two intentions in one file. Pydantic reads the tag first and then validates against
one shape, so a mistake is reported against the task that was asked for instead of as a
list of everything the tasks would between them have accepted.

One task per spec, not one per model, for the same reason there is one rank per spec:
every model shares a data pipeline, and masks and boxes are not the same pipeline.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from pyplatypus.spec.classification import ClassificationData, ClassificationModel
from pyplatypus.spec.common import SpecModel, Task
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.detection import DetectionData, DetectionModel
from pyplatypus.spec.instances import InstanceData, InstanceModel
from pyplatypus.spec.models import SegmentationModel


class PlatypusSpec(SpecModel):
    """The shared base of `SegmentationSpec` and `DetectionSpec`.

    Not built directly - `data` and `models` live on the subclasses, because their types
    are what `task` selects. `from_dict` and `from_yaml` return whichever one the task
    asks for, and both are instances of this, so anything that only needs a name, a seed
    or a rank can take a `PlatypusSpec` and not care.
    """

    task: Task = Field(
        description=(
            "What kind of problem this is: `semantic_segmentation`, "
            "`instance_segmentation` or `object_detection`. "
            "Required, and it decides the shape of everything below - `data` and `models` "
            "are validated against the task's own types, so a detection configuration "
            "without this was read as segmentation and complained about `anchors_per_grid` "
            "as an extra field instead of naming the tag. Both words are spelled out because "
            "`segmentation` stops naming one thing the moment instance segmentation exists - "
            "which it now does, so the two segmentation tasks are told apart by their tags "
            "rather than by which fields they happen to carry."
        ),
    )

    seed: int | None = Field(
        None,
        description=(
            "Makes a run repeatable as far as the arithmetic is deterministic, which goes "
            "further on some hardware than others - so it is worth knowing where it stops.\n\n"
            "**Repeatable:** the split (`split_dataset` gives the same partition on any "
            "machine), a detector's fitted anchors, the order the data arrives in and its "
            "augmentation - including inside the loader's worker processes - and, on the "
            "processor, every number to twelve decimal places.\n\n"
            "**Not repeatable:** a run on a CUDA device. Measured on a GTX 1070: two runs at "
            "one seed agreed to twelve decimals on the processor and differed in the fourth "
            "after a single epoch on the card, because cuDNN chooses algorithms this does not "
            "constrain. Nor across versions of this package - anything that changes the order "
            "data is drawn in changes what one seed draws, and `0.8.0a3` did exactly that.\n\n"
            "So it is the field for comparing two of your own runs, and not for publishing a "
            "number somebody else can land on."
        ),
    )
    deterministic: bool = Field(
        False,
        description=(
            "Ask torch for deterministic algorithms as well, which is the only thing that "
            "makes a run on a card repeatable - `seed` alone does not, and the field above "
            "says so.\n\n"
            "**It is an opt-in because the cost is real and arrives late.** Some "
            "convolutions become much slower, and an operation that has no deterministic "
            "implementation raises *when it is reached* - so a run can stop ten minutes in "
            "rather than when the specification was checked. Every other refusal here "
            "happens before anything is read; this one cannot, because whether an operation "
            "has a deterministic algorithm is a fact about the installed torch and the "
            "device, not about the configuration.\n\n"
            "Needs `seed`: constraining the algorithms while the draw still varies produces "
            "a run that is no more repeatable and only slower, so asking for it without a "
            "seed is refused rather than quietly accepted."
        ),
    )
    output_dir: str = Field(
        "platypus_output",
        description=(
            "Where a run record is written, and where `fit()` puts anything it is asked to "
            "save. Nothing is written unless something asks to write: naming this does not "
            "by itself create a directory."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def refuse_the_base_class(cls, data):
        """Say what is wrong directly. Without this, `PlatypusSpec(data=..., models=...)`
        fails as two 'extra inputs are not permitted' errors, which describes the symptom
        and not the cause."""
        if cls is PlatypusSpec:
            raise ValueError(
                "PlatypusSpec is the shared base of SegmentationSpec and DetectionSpec. "
                "Build one of those directly, or let from_dict/from_yaml choose by `task`."
            )
        return data

    @model_validator(mode="after")
    def deterministic_needs_a_seed(self):
        """Constraining the algorithms while the draw still varies buys nothing.

        `deterministic` exists so that two runs of one configuration on one card land on the
        same numbers. Without `seed` the initialisation, the shuffle and the augmentation all
        still differ, so the run is exactly as unrepeatable as before and only slower - and
        slower by an amount nobody can see the reason for. Refused here rather than warned
        about, because this is one of the two halves of the promise and a half promise is
        what platypus#122 was filed about.
        """
        if self.deterministic and self.seed is None:
            raise ValueError(
                "deterministic needs a seed: it constrains the algorithms, not the draw, so "
                "without one the run is no more repeatable and only slower"
            )
        return self

    @model_validator(mode="after")
    def unique_model_names(self):
        seen, duplicates = set(), []
        for model in self.models:
            if model.name in seen:
                duplicates.append(model.name)
            seen.add(model.name)
        if duplicates:
            raise ValueError(
                "model names must be unique; repeated: " + ", ".join(sorted(set(duplicates)))
            )
        return self

    @model_validator(mode="after")
    def one_rank_per_spec(self):
        """All models in a spec share one data pipeline, so they share one rank. Mixing
        2D and 3D in a single run is a mistake, not a feature."""
        ranks = {model.rank for model in self.models}
        if len(ranks) > 1:
            listed = ", ".join(f"{m.name}={m.rank}D" for m in self.models)
            raise ValueError(f"every model must have the same spatial rank; got {listed}")
        return self

    @property
    def rank(self) -> int:
        return self.models[0].rank

    @property
    def n_class(self) -> int:
        """How many classes the data describes. The data decides, at either task."""
        return self.data.n_class

    def check_paths(self) -> list[str]:
        return self.data.check_paths()

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the trip back across the bridge into R."""
        return self.model_dump(mode="json")


def _refuse_watching_a_number_no_split_will_produce(spec):
    """A callback cannot wait for `val_loss` in a run that has no validation set.

    The model knows which quantities it reports and the data knows whether any of them
    will be measured on a second split; neither alone can answer this, which is why it is
    here rather than beside `callbacks_watch_something_that_exists`. Without it the run
    trains to the last epoch while early stopping waits for a number that never comes, and
    model checkpointing writes nothing - silently, both of them.

    A function called by two tasks' validators rather than a method on `PlatypusSpec`,
    because detection is the third task and does not get it: that would be a change to a
    released surface made on the way past, which is how `EngineBase` diverged twice. The
    two that share it share it from one place; the third is a decision of its own.
    """
    if spec.data.validation:
        return spec
    offenders = [
        f"{model.name}: {callback.name} watches '{callback.monitor}'"
        for model in spec.models
        for callback in model.callbacks
        if getattr(callback, "monitor", None) is not None
        and str(callback.monitor).startswith("val_")
    ]
    if offenders:
        listed = "\n  - ".join(offenders)
        raise ValueError(
            f"`validation: false` means no `val_` number is ever produced, and these "
            f"would wait for one:\n  - {listed}\nWatch the training quantity instead, "
            f"or give the run a validation set."
        )
    return spec


class SegmentationSpec(PlatypusSpec):
    task: Literal[Task.SEMANTIC_SEGMENTATION] = Field(
        Task.SEMANTIC_SEGMENTATION,
        description="Fixed for this specification: `semantic_segmentation`.",
    )
    data: SegmentationData = Field(
        description="Where the images and their masks are, and how to read them."
    )
    models: list[SegmentationModel] = Field(
        min_length=1,
        description=(
            "One or more models, trained in turn against the same data. Names must be "
            "unique: a run is recorded under its model's name, so two models called the same "
            "thing would overwrite one record and report it as two. Several entries of the "
            "same architecture with different settings is the ordinary case - that is how a "
            "comparison is written."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def channels_follow_the_data(cls, config):
        """Fill `channels` from `channels_from` before the models are built.

        Before rather than after, because a built model is frozen - and frozen is right: a
        specification that rewrote itself during validation would be a different thing from
        the one somebody wrote. This fills a blank in the configuration instead, which is
        what a reader would have had to do by hand.
        """
        if not isinstance(config, dict):
            return config
        data = config.get("data")
        if not isinstance(data, dict):
            return config
        patterns = data.get("channels_from")
        if not patterns:
            return config
        models = config.get("models")
        if not isinstance(models, list):
            return config
        config = {
            **config,
            "models": [
                {**m, "channels": len(patterns)}
                if isinstance(m, dict) and "channels" not in m
                else m
                for m in models
            ],
        }
        return config

    @model_validator(mode="after")
    def overlapping_classes_rule_out_two_settings(self):
        """What an independent-channel naming cannot be combined with, said here rather
        than at training time.

        Both of them: `regions` and `classes_from` differ only in the file on disk, so the
        refusals read `data.multilabel` and name whichever field was written. The prose
        below says `regions` because that is the naming it was found on.

        Two things, and they fail differently.

        **`lovasz`** is not a tuning away from working. The Lovasz-Softmax surrogate sorts
        the errors of a softmax over mutually exclusive classes and walks the Lovasz
        extension of the Jaccard loss along that order; the independent form is the
        Lovasz-Hinge, a different construction. Run as it stands on sigmoid channels it
        would produce a number and a gradient, which is the worst available outcome.

        **`include_background: false`** has nothing to exclude: `regions` has no background
        entry, so channel 0 is the first region somebody asked for and dropping it would
        drop a class. Refused only when it was *written* - `cldice` defaults it to False, and
        a default is not a request. `model_fields_set` is what tells those apart, the same
        way `channels_match_the_data` tells a stated `channels` from the one that was there
        anyway.
        """
        if not self.data.multilabel:
            return self
        offenders = [
            f"{model.name}: loss {model.loss.name}"
            for model in self.models
            if not model.loss.supports_multilabel
        ]
        if offenders:
            listed = "\n  - ".join(offenders)
            raise ValueError(
                f"`{self.data.multilabel_naming}` makes each channel an independent "
                "question, and these losses are defined over mutually exclusive "
                f"classes:\n  - {listed}\n"
                "Lovasz-Softmax would still return a number on sigmoid channels, which is "
                "why this is refused rather than warned about. Use dice, iou, cce, "
                "cce_dice, focal, tversky, focal_tversky, combo or boundary."
            )
        excluding = [
            f"{model.name}: metric {metric.name}"
            for model in self.models
            for metric in model.metrics
            if "include_background" in metric.model_fields_set and not metric.include_background
        ]
        if excluding:
            listed = "\n  - ".join(excluding)
            naming = self.data.multilabel_naming
            raise ValueError(
                f"`include_background: false` asks to leave out a class `{naming}` does "
                f"not have:\n  - {listed}\nWith {naming} there is no background entry, so "
                "channel 0 is the first class named and dropping it would drop a real one. "
                f"Leave the setting out; with {naming} it is ignored rather than honoured."
            )
        return self

    @model_validator(mode="after")
    def nothing_watches_a_validation_number_that_will_not_arrive(self):
        """A callback cannot wait for `val_loss` in a run that has no validation set."""
        return _refuse_watching_a_number_no_split_will_produce(self)

    @model_validator(mode="after")
    def channels_match_the_data(self):
        """One pattern per channel - **derived when the model stayed silent, checked when it
        did not**.

        `channels_from` names one file per channel, so it already says how many there are;
        a model restating it could only ever disagree, and the disagreement surfaced deep
        inside torch as a shape mismatch rather than here. Unlike `n_class` the field does
        not simply go: without `channels_from` it is a real choice, because the same files
        can be read as one channel or three.

        So: silent means derived, stated means checked. `model_fields_set` is what tells
        those apart - a `channels=3` the caller wrote and the default that was there anyway
        are the same value and not the same claim.
        """
        if self.data.channels_from is None:
            return self
        expected = len(self.data.channels_from)
        wrong = [
            f"{model.name} has channels={model.channels}"
            for model in self.models
            if "channels" in model.model_fields_set and model.channels != expected
        ]
        if wrong:
            raise ValueError(f"channels_from lists {expected} channels, but " + "; ".join(wrong))
        return self


class InstanceSpec(PlatypusSpec):
    """Objects rather than classes: which object every pixel belongs to, scored as instances.

    The same images and the same U-shaped networks as `semantic_segmentation`, with the
    masks read one file per object instead of united into one. Two fields are new on a
    model - `separation`, what the network predicts, and `decode`, a list of ways to turn
    those channels into objects - and `pyplatypus.spec.instances` is where both are
    documented, measurements included.

    **What this task is honestly worth, said here because the tag's name promises more.**
    A reader who writes `task: instance_segmentation` gets a semantic model with a
    separation target, decoded into objects and scored by the Data Science Bowl's own
    average precision. On that dataset the separation signal is worth +0.068 at best while
    a perfect mask is worth +0.2398, so the limit is the mask rather than the separation -
    and `connected_components`, which ignores the signal entirely, is always available as
    the row every other row is read against. The capability is real, the ceiling is
    measured, and both are in the documentation rather than in a footnote.
    """

    task: Literal[Task.INSTANCE_SEGMENTATION] = Field(
        Task.INSTANCE_SEGMENTATION,
        description="Fixed for this specification: `instance_segmentation`.",
    )
    data: InstanceData = Field(
        description="Where the images and their per-object masks are, and how to read them."
    )
    models: list[InstanceModel] = Field(
        min_length=1,
        description=(
            "One or more models, trained in turn against the same data. Names must be "
            "unique, for the same reason as in a segmentation specification. Several "
            "entries differing only in `separation` is the comparison this task exists to "
            "make easy - and `decode` widens the table again without a second run."
        ),
    )

    @model_validator(mode="after")
    def nothing_watches_a_validation_number_that_will_not_arrive(self):
        """A callback cannot wait for `val_loss` in a run that has no validation set."""
        return _refuse_watching_a_number_no_split_will_produce(self)


class ClassificationSpec(PlatypusSpec):
    """One label for a whole image, scored by how well the answers can be ranked.

    The task `crops()` has been cutting objects out for and handing them to nothing, and the
    one explaining a model needs: a saliency map explains *this answer*, and until now there
    was no per-image answer to explain.

    A contracting path, pooled, with one dense layer on top - ours or any of timm's, chosen
    by `encoder`, which is why this is the one task with no `architecture` field: there is
    nothing after the pooling for a second name to describe.

    **What is worth knowing before writing one, because it decides what the numbers mean.**
    The default metric is `auc` rather than `accuracy`, measured rather than preferred: on
    40000 images at a prevalence of 3%, with the model's own probabilities as the truth so
    that only its discrimination varies, accuracy spans 0.0100 where the AUC spans 0.4631
    and is **not monotone** - a model with no signal at all scores above two models that
    have one. Balanced accuracy and F1 do not rescue it either, because a calibrated model
    at that prevalence never crosses the threshold and everything read off the hard
    prediction sees the same all-negative answer.

    The same arithmetic decides the split. A random 70/15/15 of 100 images at 3% prevalence
    leaves some part with none of the rare class 91% of the time, so `split` is stratified
    for this task and a class with fewer examples than there are parts is refused by name.
    `ClassificationEngine.balance()` reports the prevalence rather than leaving it to be
    discovered from a score that looks fine.
    """

    task: Literal[Task.CLASSIFICATION] = Field(
        Task.CLASSIFICATION,
        description="Fixed for this specification: `classification`.",
    )
    data: ClassificationData = Field(
        description=(
            "Where the images are and what their labels are called. The label is the "
            "directory an image sits in, or a `label` column naming the class."
        )
    )
    models: list[ClassificationModel] = Field(
        min_length=1,
        description=(
            "One or more models, trained in turn against the same data. Names must be "
            "unique, for the same reason as in a segmentation specification. Several "
            "entries differing only in `encoder` is the comparison this task makes easiest "
            "- ours against a pretrained backbone, on the same split."
        ),
    )

    @model_validator(mode="after")
    def nothing_watches_a_validation_number_that_will_not_arrive(self):
        """A callback cannot wait for `val_loss` in a run that has no validation set."""
        return _refuse_watching_a_number_no_split_will_produce(self)


class DetectionSpec(PlatypusSpec):
    task: Literal[Task.OBJECT_DETECTION] = Field(
        description="Fixed for this specification: `object_detection`.",
    )
    data: DetectionData = Field(
        description="Where the images and their annotations are, and how to read them."
    )
    models: list[DetectionModel] = Field(
        min_length=1,
        description=(
            "One or more detectors, trained in turn against the same data. Names must be "
            "unique, for the same reason as in a segmentation specification."
        ),
    )


#: Validate through this, never through one class, so `task` picks the shape.
AnySpec = Annotated[
    SegmentationSpec | InstanceSpec | DetectionSpec | ClassificationSpec,
    Field(discriminator="task"),
]

SPEC_ADAPTER: TypeAdapter[SegmentationSpec | InstanceSpec | DetectionSpec | ClassificationSpec] = (
    TypeAdapter(AnySpec)
)
