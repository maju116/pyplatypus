"""The whole experiment, and the single contract between R and Python.

Both `platypus_spec(...)` in R and a YAML file produce this object. Nothing downstream
can tell which one was used, which is the entire point.

`task` chooses between two of them. It is a discriminator rather than a setting because
it decides the *type* of `data` and of every entry in `models`: a detection spec holding
a `colormap`, or a segmentation spec holding `anchors`, is not a spec with a stray field
but two intentions in one file. Pydantic reads the tag first and then validates against
one shape, so a mistake is reported against the task that was asked for instead of as a
list of everything both tasks would have accepted.

One task per spec, not one per model, for the same reason there is one rank per spec:
every model shares a data pipeline, and masks and boxes are not the same pipeline.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from pyplatypus.spec.common import SpecModel, Task
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.detection import DetectionData, DetectionModel
from pyplatypus.spec.models import SegmentationModel


class PlatypusSpec(SpecModel):
    """The shared base of `SegmentationSpec` and `DetectionSpec`.

    Not built directly - `data` and `models` live on the subclasses, because their types
    are what `task` selects. `from_dict` and `from_yaml` return whichever one the task
    asks for, and both are instances of this, so anything that only needs a name, a seed
    or a rank can take a `PlatypusSpec` and not care.
    """

    task: Task

    seed: int | None = Field(None, description="Set it if you want a reproducible run.")
    output_dir: str = "platypus_output"

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


class SegmentationSpec(PlatypusSpec):
    task: Literal[Task.SEMANTIC_SEGMENTATION] = Task.SEMANTIC_SEGMENTATION
    data: SegmentationData
    models: list[SegmentationModel] = Field(min_length=1)

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
    def nothing_watches_a_validation_number_that_will_not_arrive(self):
        """A callback cannot wait for `val_loss` in a run that has no validation set.

        The model knows which quantities it reports and the data knows whether any of them
        will be measured on a second split; neither alone can answer this, which is why it
        is here rather than beside `callbacks_watch_something_that_exists`. Without it the
        run trains to the last epoch while early stopping waits for a number that never
        comes, and model checkpointing writes nothing - silently, both of them.
        """
        if self.data.validation:
            return self
        offenders = [
            f"{model.name}: {callback.name} watches '{callback.monitor}'"
            for model in self.models
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
        return self

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


class DetectionSpec(PlatypusSpec):
    task: Literal[Task.OBJECT_DETECTION]
    data: DetectionData
    models: list[DetectionModel] = Field(min_length=1)


#: Validate through this, never through one class, so `task` picks the shape.
AnySpec = Annotated[SegmentationSpec | DetectionSpec, Field(discriminator="task")]

SPEC_ADAPTER: TypeAdapter[SegmentationSpec | DetectionSpec] = TypeAdapter(AnySpec)
