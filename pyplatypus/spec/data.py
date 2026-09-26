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

from pydantic import AliasChoices, Field, field_validator, model_validator

from pyplatypus.spec.common import WINDOWS, DataMode, SpecModel

Colour = Annotated[tuple[int, int, int], Field(description="RGB, each channel 0-255.")]


class SegmentationData(SpecModel):
    train_path: str
    validation_path: str
    test_path: str | None = None

    mode: DataMode = DataMode.NESTED_DIRS
    colormap: list[Colour] | None = Field(
        None,
        min_length=2,
        description="One colour per class; index in this list is the class index.",
    )
    labels: list[int] | None = Field(
        None,
        min_length=2,
        description=(
            "For masks stored as label maps rather than pictures: the voxel value of each "
            "class, in class order. This is how volumes label anything - NIfTI holds "
            "integers, not colours - and how a single-channel PNG mask can be read too. "
            "Exactly one of `colormap` and `labels` is given."
        ),
    )
    window: str | tuple[float, float] = Field(
        "auto",
        validation_alias=AliasChoices("window", "dicom_window"),
        description=(
            "How values in real units are mapped to 0-1: a named window such as 'lung' or "
            "'soft_tissue', an explicit (centre, width) pair, 'auto' for the window the "
            "file recorded - DICOM only, since NIfTI stores none - or 'full' for the whole "
            "range present. A fixed window is what makes two scans comparable: scaling each "
            "one by its own extremes lets a single bright voxel rescale everything else. "
            "Ignored for ordinary pictures, which are already 0-255. Accepted as "
            "`dicom_window` too, the name it had while DICOM was the only format that "
            "needed it."
        ),
    )
    channels_from: list[str] | None = Field(
        None,
        min_length=2,
        description=(
            "For datasets that keep one channel per file - BraTS ships T1, T1ce, T2 and FLAIR "
            "per patient; 38-Cloud keeps its bands apart - one pattern per channel, in channel "
            "order, each a regular expression matched against the file names. Every pattern "
            "must match exactly one of a sample's files. Stated rather than inferred because "
            "sorting gives flair, t1, t1ce, t2: reproducible, and anatomically meaningless. A "
            "model trained with FLAIR in channel one and used on data where channel one is T1 "
            "returns a plausible answer and no error."
        ),
    )
    target_spacing: tuple[float, float, float] | None = Field(
        None,
        description=(
            "Resample every volume to this many millimetres per voxel before training, then "
            "centre-crop or pad to the model's input_shape. Without it a volume is simply "
            "resized to input_shape, which gives two scans of the same anatomy different "
            "physical scale when they were acquired at different slice thicknesses - and the "
            "network has no way to know. Volumes only; ignored for 2D."
        ),
    )
    subdirs: tuple[str, str] = ("images", "masks")
    column_sep: str = ";"
    shuffle: bool = True

    @field_validator("window")
    @classmethod
    def known_window(cls, value):
        if isinstance(value, str) and value not in {"auto", "full"} and value not in WINDOWS:
            raise ValueError(
                f"unknown window '{value}'; use 'auto', 'full', a (centre, width) pair, "
                f"or one of: {', '.join(sorted(WINDOWS))}"
            )
        if not isinstance(value, str):
            width = value[1]
            if width <= 0:
                raise ValueError(f"window width must be positive, got {width}")
        return value

    @field_validator("colormap")
    @classmethod
    def channels_in_range(cls, value: list[tuple[int, int, int]] | None):
        if value is None:
            return value
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

    @field_validator("channels_from")
    @classmethod
    def patterns_are_distinct(cls, value):
        if value is not None and len(set(value)) != len(value):
            raise ValueError(
                "channel patterns must be distinct - two channels matching the same file "
                "would make one measurement into two"
            )
        return value

    @field_validator("target_spacing")
    @classmethod
    def spacing_is_positive(cls, value):
        if value is not None and any(v <= 0 for v in value):
            raise ValueError(f"target_spacing must be positive millimetres, got {value}")
        return value

    @field_validator("labels")
    @classmethod
    def labels_are_distinct(cls, value: list[int] | None):
        if value is None:
            return value
        if len(set(value)) != len(value):
            raise ValueError(
                "label values must be distinct - two classes sharing a value cannot be "
                "told apart"
            )
        return value

    @model_validator(mode="after")
    def one_way_of_naming_classes(self):
        """Exactly one of `colormap` and `labels`.

        Not both, even when they agree: two sources for the number of classes is two places
        to change it and one of them will be forgotten. Not neither, because nothing else in
        the spec says how many classes a mask holds - and guessing it from the data would
        mean a dataset whose validation set happens to contain no tumour trains a model with
        one class fewer.
        """
        if (self.colormap is None) == (self.labels is None):
            raise ValueError(
                "give exactly one of `colormap` (masks stored as pictures) and `labels` "
                "(masks stored as label maps, which is how volumes do it)"
            )
        return self

    @property
    def n_class(self) -> int:
        """How many classes there are. The colormap or the labels decide; nothing else."""
        return len(self.colormap if self.colormap is not None else self.labels)

    @property
    def label_map(self) -> bool:
        """Whether masks are label maps rather than pictures."""
        return self.labels is not None

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
