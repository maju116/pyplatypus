"""The whole experiment, and the single contract between R and Python.

Both `platypus_spec(...)` in R and a YAML file produce this object. Nothing downstream
can tell which one was used, which is the entire point.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from pyplatypus.spec.common import SpecModel
from pyplatypus.spec.data import SegmentationData
from pyplatypus.spec.models import SegmentationModel


class PlatypusSpec(SpecModel):
    data: SegmentationData
    models: list[SegmentationModel] = Field(min_length=1)

    seed: int | None = Field(None, description="Set it if you want a reproducible run.")
    output_dir: str = "platypus_output"

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

    @property
    def rank(self) -> int:
        return self.models[0].rank

    def check_paths(self) -> list[str]:
        return self.data.check_paths()

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the trip back across the bridge into R."""
        return self.model_dump(mode="json")
