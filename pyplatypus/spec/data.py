"""Where the images and masks come from.

Note what is *not* here: existence checks on paths. The old package raised
`NotADirectoryError` inside a validator, which meant a spec could not be built unless
the data was already sitting on that machine - awkward for tests, and impossible if R
builds a spec to hand somewhere else. Paths are checked by `check_paths()`, a separate
step the loader runs by default and a caller can skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from pyplatypus.spec.common import DataMode, SpecModel

Colour = Annotated[tuple[int, int, int], Field(description="RGB, each channel 0-255.")]


class SegmentationData(SpecModel):
    train_path: str
    validation_path: str
    test_path: str | None = None

    mode: DataMode = DataMode.NESTED_DIRS
    colormap: list[Colour] = Field(
        min_length=2,
        description="One colour per class; index in this list is the class index.",
    )
    subdirs: tuple[str, str] = ("images", "masks")
    column_sep: str = ";"
    shuffle: bool = True

    @field_validator("colormap")
    @classmethod
    def channels_in_range(cls, value: list[tuple[int, int, int]]):
        for index, colour in enumerate(value):
            if any(channel < 0 or channel > 255 for channel in colour):
                raise ValueError(
                    f"colour {index} is {colour}; every channel must be between 0 and 255"
                )
        if len(set(value)) != len(value):
            raise ValueError(
                "colours must be distinct - two classes sharing a colour cannot be told apart"
            )
        return value

    @property
    def n_class(self) -> int:
        """The colormap decides how many classes there are. Nothing else gets a vote."""
        return len(self.colormap)

    def check_paths(self) -> list[str]:
        """Return a human-readable problem for every path that is not there."""
        problems = []
        wanted = [("train_path", self.train_path), ("validation_path", self.validation_path)]
        if self.test_path is not None:
            wanted.append(("test_path", self.test_path))
        for field, value in wanted:
            path = Path(value)
            if not path.exists():
                problems.append(f"{field}: '{value}' does not exist")
            elif self.mode is DataMode.NESTED_DIRS and not path.is_dir():
                problems.append(f"{field}: '{value}' is not a directory, but mode is nested_dirs")
            elif self.mode is DataMode.CONFIG_FILE and not path.is_file():
                problems.append(f"{field}: '{value}' is not a file, but mode is config_file")
        return problems
