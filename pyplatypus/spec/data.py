"""Where the data comes from.

`DataSpec` holds what every task needs - the three paths, the layout, the window - and a
task's own class adds what only it needs. Segmentation needs to be told how a mask names
its classes; detection needs to be told how an annotation file does. Neither question has
an answer that fits the other, which is why there are two classes rather than one with
optional halves.

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


class SplitSpec(SpecModel):
    """How to divide one folder, when there is only one folder.

    The alternative to naming `validation_path`: a researcher with a single directory
    should not have to run a splitting tool and then point three paths at its output,
    and nothing about that round trip was ever checked.

    Nothing is written. `split_samples` partitions the samples that were already
    discovered, so a specification stays a description and gains no side effect on disk.
    `split_dataset()` is still the tool when you want the CSVs to keep, to hand to a
    colleague or to cite.
    """

    fractions: tuple[float, ...] = Field(
        description=(
            "Two or three: train, validation, and optionally test. The length rule is "
            "enforced by the splitting module rather than restated here - one rule, one "
            "place. A third fraction is scoreable, because it comes from annotated data."
        ),
    )

    group_by: str | None = Field(
        description=(
            "**Required, and may be null.** A regular expression read against the sample "
            "key; everything sharing a group lands in exactly one split. Usually a patient, "
            "sometimes a study, a scanner or a site - whatever must not straddle the "
            "division.\n\n"
            "Required rather than optional because the mistake it prevents is invisible. "
            "Slices of one patient in training and validation at once make validation "
            "measure memory instead of generalisation, and Dice comes out several points too "
            "high with nothing in the output to say so. An omitted field would be that "
            "choice made silently; `null` is the same choice made on purpose, which is all "
            "this asks for."
        ),
    )

    seed: int = Field(
        0,
        description=(
            "The split is deterministic given the samples, the fractions, the pattern and "
            "this. A split that moves between runs makes two results incomparable and the "
            "reason for a difference impossible to find."
        ),
    )

    @model_validator(mode="after")
    def fractions_are_two_or_three(self):
        from pyplatypus.data.splits import _fractions

        _fractions(self.fractions)  # raises with the message that module already gives
        return self


class DataSpec(SpecModel):
    """The part of "where is the data" that does not depend on what is being learned."""

    train_path: str = Field(
        description=(
            "Where the training data is. For `nested_dirs`, a directory of sample "
            "directories; for `config_file`, a CSV naming the files. With a `split`, this is "
            "the only path given and the divisions are cut from it."
        ),
    )
    validation_path: str | None = Field(
        None,
        description=(
            "Optional only in the sense that `split` is the other way to get one, and "
            "`validation: false` the third. Exactly one of them, because two sources for the "
            "validation set are two places to change it."
        ),
    )
    test_path: str | None = Field(
        None,
        description=(
            "A third set, scored only when asked. Whether it can be scored at all depends on "
            "whether it carries labels: a folder of images alone is accepted and `predict` "
            "works on it, while `evaluate` refuses it by name rather than failing two layers "
            "down on the first missing mask."
        ),
    )
    validation: bool = Field(
        True,
        description=(
            "Whether this run validates at all. `false` is the third and last way of "
            "answering the question `validation_path` and `split` answer, and it has to be "
            "written down.\n\n"
            "A final fit on every case you have, once the hyperparameters are settled, is a "
            "legitimate thing to want - and until now it meant inventing a split and "
            "ignoring the number it produced. What is *not* legitimate is arriving here by "
            "omission: a specification with neither a path nor a split is still an error, "
            "because 'I have no validation set' and 'I forgot' look identical and only one "
            "of them is a decision.\n\n"
            "With `false`, giving `validation_path` or `split` as well is refused: they say "
            "where the validation set comes from and this says there is not one."
        ),
    )

    split: SplitSpec | None = Field(
        None,
        description=(
            "Divide `train_path` instead of naming a second folder. Nothing is written: the "
            "samples that were already discovered are partitioned, so a specification stays "
            "a description and leaves no files behind. `split_dataset()` is the tool when the "
            "CSVs are the point - to keep, to hand to a colleague, or to cite."
        ),
    )

    mode: DataMode = Field(
        DataMode.NESTED_DIRS,
        description=(
            "How the files are arranged: `nested_dirs` for one directory per sample, "
            "`config_file` for a CSV whose columns name the files. The CSV form is what to "
            "use when the data cannot be moved or copied into a layout."
        ),
    )
    subdirs: tuple[str, str] = Field(
        description=(
            "For `nested_dirs`, the two subdirectories of each sample: the images, and "
            "whatever labels them - masks for segmentation, annotation files for "
            "detection. No default here; each task has its own."
        ),
    )
    column_sep: str = Field(
        ";",
        description=(
            "For `config_file`, what separates several paths inside one cell - a sample with "
            "one mask file per object, as Data Science Bowl has, or one file per channel."
        ),
    )
    shuffle: bool = Field(
        True,
        description=(
            "Shuffle the training data between epochs. Only the training data: validation is "
            "read in order so that two epochs measure the same thing in the same way."
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

    @model_validator(mode="after")
    def one_way_of_getting_a_validation_set(self):
        """Exactly one of `validation_path`, `split`, and `validation: false`.

        Not neither: a run with nothing to validate against reports a number that
        describes the training data, which is the most dangerous number this package
        could make easy to produce. That stays true, which is why the third option is a
        field someone has to write rather than silence being allowed to mean it.

        Not two of them, for the reason `colormap` and `labels` are not both: two
        sources for one thing are two places to change it and one gets forgotten.
        """
        if not self.validation:
            named = [
                n
                for n, v in (("validation_path", self.validation_path), ("split", self.split))
                if v is not None
            ]
            if named:
                joined = " and ".join("`" + n + "`" for n in named)
                raise ValueError(
                    f"`validation: false` says this run has no validation set, and "
                    f"{joined} says where it comes from. Drop one: they cannot both be true."
                )
        elif (self.validation_path is None) == (self.split is None):
            raise ValueError(
                "a run needs something to validate against: give `validation_path`, or "
                "`split` to divide `train_path` itself - and `split.group_by` keeps a "
                "patient out of both halves, which dividing by file does not. To train on "
                "everything and validate on nothing, say `validation: false` - it is a "
                "decision and has to be written down rather than arrived at by omission"
            )
        if self.split is not None and self.test_path is not None:
            raise ValueError(
                "`test_path` and `split` both say where the test set comes from; give the "
                "third fraction to `split.fractions` instead, or drop it"
            )
        return self

    @property
    def label_column(self) -> str:
        """The CSV column holding whatever labels an image.

        `config_file` mode has to call it something, and the two tasks label differently.
        Derived rather than a field: one more name to get wrong, for a value that follows
        from the task.
        """
        return "masks"

    def check_paths(self) -> list[str]:
        """Return a human-readable problem for every path that is not there."""
        problems = []
        wanted = [("train_path", self.train_path)]
        # Absent when `split` divides the training folder instead, which the validator
        # above has already established is one or the other.
        if self.validation_path is not None:
            wanted.append(("validation_path", self.validation_path))
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


class SegmentationData(DataSpec):
    subdirs: tuple[str, str] = Field(
        ("images", "masks"),
        description=(
            "For `nested_dirs`, the two subdirectories of each sample directory: the images "
            "and their masks. A sample may hold several mask files - one per object, as Data "
            "Science Bowl does - and they are united into one mask when read."
        ),
    )

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
                "label values must be distinct - two classes sharing a value cannot be told apart"
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
