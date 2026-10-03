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
    task: Literal[Task.SEGMENTATION] = Task.SEGMENTATION
    data: SegmentationData
    models: list[SegmentationModel] = Field(min_length=1)

    @model_validator(mode="after")
    def channels_match_the_data(self):
        """One pattern per channel, or the stack the data produces is not the one the model
        declared - which fails deep inside torch with a shape mismatch instead of here."""
        if self.data.channels_from is None:
            return self
        expected = len(self.data.channels_from)
        wrong = [
            f"{model.name} has channels={model.channels}"
            for model in self.models
            if model.channels != expected
        ]
        if wrong:
            raise ValueError(
                f"channels_from lists {expected} channels, but " + "; ".join(wrong)
            )
        return self

    @model_validator(mode="after")
    def classes_match_the_data(self):
        expected = self.data.n_class
        wrong = [
            f"{model.name} has n_class={model.n_class}"
            for model in self.models
            if model.n_class != expected
        ]
        if wrong:
            # Named after whichever one is in use, because "the colormap defines 3 classes"
            # is a confusing thing to be told about a spec that has no colormap.
            source = "labels" if self.data.label_map else "colormap"
            raise ValueError(
                f"the {source} defines {expected} classes, but " + "; ".join(wrong)
            )
        return self


class DetectionSpec(PlatypusSpec):
    task: Literal[Task.DETECTION]
    data: DetectionData
    models: list[DetectionModel] = Field(min_length=1)


#: Validate through this, never through one class, so `task` picks the shape.
AnySpec = Annotated[SegmentationSpec | DetectionSpec, Field(discriminator="task")]

SPEC_ADAPTER: TypeAdapter[SegmentationSpec | DetectionSpec] = TypeAdapter(AnySpec)
