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


class Region(SpecModel):
    """One scored class, named as a union of the values a label map already holds.

    The case this exists for is the one where overlapping classes are a *convention applied
    to exclusive data* rather than a different file format. BraTS gives every voxel one of
    `{0, 1, 2, 4}` and scores three regions that are unions of those values:

        enhancing tumour  of: [4]
        tumour core       of: [1, 4]
        whole tumour      of: [1, 2, 4]

    The same shape appears wherever a lesion sits inside an organ - liver and tumour, kidney
    and cyst, lung and nodule - and nothing new has to be read off disk for it.

    `of` is a membership test, which is why nothing here has to declare the full vocabulary
    of values the mask holds: a value no region mentions is simply absent from every channel.
    That is also the mistake this invites, so `inspect_masks()` counts what each region
    matched - a region naming a value the data does not hold is a channel of zeros, and a
    channel of zeros trains to predict nothing and scores a Dice of 0 without complaining.
    """

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "What this region is called. It names a channel of the output, so it is what a "
            "per-class score is reported under and what `save_class_masks` puts in a "
            "filename. Distinct from every other region's name."
        ),
    )
    of: list[int] = Field(
        ...,
        min_length=1,
        description=(
            "The label-map values belonging to this region, in any order. A voxel is in the "
            "region when its value is one of these, so regions may overlap - that is the "
            "point - and a single value is normal: BraTS's enhancing tumour is `of: [4]`. "
            "Disjoint regions are allowed and mean something useful, grouping several "
            "labels into one class and dropping the background channel; they are not "
            "a mistake and are not refused."
        ),
    )

    @field_validator("of")
    @classmethod
    def values_are_distinct(cls, value: list[int]):
        if len(set(value)) != len(value):
            raise ValueError(
                f"a region lists {value}; a repeated value says nothing twice and is "
                "more likely a typo than an intention"
            )
        return value


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

    def measurement_fingerprint(self) -> dict:
        """How the data was READ, which decides what a reported figure was measured on.

        The data's half of a sidecar's `measured_with`. `ModelSpec.measurement_fingerprint`
        is the model's, and its first sentence says it is about what a figure depends on
        *beyond the weights and the data* - so these fields cannot go there without making
        that sentence false, and `pyplatypus.weights` is handed a model and never sees a data
        spec at all. The engine joins the two, the way it already contributes
        `_class_fingerprint`.

        `pyplatypus#447`. `#427` recorded `splits` because a source image is read at
        `splits * input_shape` and cut, so the tiling changes the pictures a metric sees
        while the weights and the data stay the same. Three fields one layer out pass that
        same test and are the data's rather than the model's:

            window          soft_tissue against lung on pydicom's own `CT_small.dcm` moves
                            a picture by 0.793 of its range; against the full range,
                            0.449, 0.493 and 0.192 for soft tissue, lung and bone
            channels_from   `measurements/braintumour/channel-order.json` scores ONE
                            trained model at Dice 0.467 against 0.110 on a permuted order
            target_spacing  a 24-voxel cube at source spacing (1, 1, 3) resamples to
                            (24, 24, 72), (24, 24, 24) or (12, 12, 36) for targets
                            (1,1,1), (1,1,3) and (2,2,2) - 18432, 13824 and 4608 real
                            voxels at an IDENTICAL model input shape

        **Only what is set**, which is the author's decision of 2026-11-08 and
        `_class_fingerprint`'s own shape, since that writes the one naming that is there.
        Three `null`s would say *measured without a window*, and that is false of a file an
        older engine wrote because it could not answer. `window` is therefore always here:
        it always has a value, and `'auto'` is a windowing rather than an absence.

        **Resolved rather than as written**, for the reason the model's half gives in full:
        the value that made `#413` a day of measurement was a default nobody typed.

        What this does NOT claim is that every data field a figure depends on is recorded.
        `split`, `validation` and `shuffle` decide which cases were held out, and all five
        published sidecars carry `split` because each publisher writes it by hand - which is
        `#419`'s argument about `trained_with` one key over. `pyplatypus#450` is that
        question with its three options and a recommendation, so this is a remainder rather
        than a silence.
        """
        # A named window is a string and an explicit one is a (centre, width) tuple, which
        # `json.dump` would write as a list anyway. Converted here so that what this returns
        # is already what the file will hold, rather than something that survives one
        # serialiser and not another.
        window = self.window if isinstance(self.window, str) else list(self.window)
        return {"window": window}


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
            "Exactly one of `colormap` and `labels` is given. **Include 0 unless the masks "
            "really do fill every voxel**: a value matching no entry falls back to the "
            "first class, so a list without 0 trains the background as whatever that first "
            "class is, and `fit` refuses the combination rather than let it look plausible."
        ),
    )
    regions: list[Region] | None = Field(
        None,
        min_length=2,
        description=(
            "Overlapping classes, named as unions of a label map's values - the third way of "
            "saying what a mask's classes are, and the only one that lets a voxel belong to "
            "more than one. Each entry gives a `name` and the values it is made `of`. With "
            "`regions` the model's channels are independent binary questions: a sigmoid per "
            "channel rather than one softmax across them, and **no background entry**, for "
            "the same reason detection's `classes` has none. Exactly one of `colormap`, "
            "`labels` and `regions` is given. Two regions are the minimum - one region is "
            "ordinary binary segmentation, which `labels: [0, 1]` already says."
        ),
    )
    classes_from: list[str] | None = Field(
        None,
        min_length=2,
        description=(
            "For masks kept as one binary file per class - the fourth way of saying what a "
            "mask's classes are, and the only one that can express classes which overlap "
            "*independently*: a vessel crossing an organ, or two annotators' masks of the "
            "same structure. One pattern per class, in class order, each a regular "
            "expression matched against a sample's mask file names, and every pattern must "
            "match exactly one file - the rules `channels_from` uses, through the same "
            "matcher.\n\n"
            "Stated rather than inferred for the reason `channels_from` gives, with worse "
            "consequences: sorting the names is reproducible and anatomically meaningless, "
            "and a model trained with 'tumour' in channel 1 and used where channel 1 is "
            "'oedema' answers plausibly, in range, and nothing downstream can tell.\n\n"
            "Like `regions` this makes every channel an independent binary question - a "
            "sigmoid per channel rather than one softmax across them - and has **no "
            "background entry**. The difference between the two is the file on disk and "
            "nothing else: `regions` derives overlap from one label map's values, which is "
            "what BraTS and the liver-and-tumour family need, and this reads overlap that "
            "was drawn separately in the first place. Exactly one of `colormap`, `labels`, "
            "`regions` and `classes_from` is given."
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

    @field_validator("channels_from", "classes_from")
    @classmethod
    def patterns_are_distinct(cls, value, info):
        """One validator for both, because the rule and the reason are the same one.

        `info.field_name` is what keeps the message about the field somebody wrote rather
        than about channels in a message printed for classes.
        """
        noun = "channel" if info.field_name == "channels_from" else "class"
        if value is not None and len(set(value)) != len(value):
            raise ValueError(
                f"{noun} patterns must be distinct - two {noun}es matching the same file "
                f"would make one {noun} into two"
                if noun == "class"
                else "channel patterns must be distinct - two channels matching the same "
                "file would make one measurement into two"
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

    @field_validator("regions")
    @classmethod
    def regions_are_distinct(cls, value: list[Region] | None):
        if value is None:
            return value
        names = [region.name for region in value]
        if len(set(names)) != len(names):
            raise ValueError(
                f"region names must be distinct, got {names} - a name identifies a channel "
                "of the output and a per-class score is reported under it"
            )
        sets = [frozenset(region.of) for region in value]
        if len(set(sets)) != len(sets):
            repeated = next(s for s in sets if sets.count(s) > 1)
            raise ValueError(
                f"two regions are both made of {sorted(repeated)}; they would be two "
                "channels asking one question, which makes one measurement into two"
            )
        return value

    @model_validator(mode="after")
    def one_way_of_naming_classes(self):
        """Exactly one of `colormap`, `labels` and `regions`.

        Not two of them, even when they agree: two sources for the number of classes is two
        places to change it and one of them will be forgotten. Not none, because nothing else
        in the spec says how many classes a mask holds - and guessing it from the data would
        mean a dataset whose validation set happens to contain no tumour trains a model with
        one class fewer.

        `regions` is the third and it reads the same file `labels` does, which is why it
        replaces it rather than joining it: a region is a membership test over the values in
        a label map, so the complete list of those values never has to be written down.

        `classes_from` is the fourth and it is the only one that is about the *files*: one
        binary mask per class, which is the only shape that can hold two things drawn
        independently over the same voxel. It excludes the other three for the same reason
        they exclude each other - it answers how many classes there are and what each one
        is - and in particular it is not a modifier on `labels`: a per-class file has no
        label values to list.
        """
        given = [
            name
            for name, value in (
                ("colormap", self.colormap),
                ("labels", self.labels),
                ("regions", self.regions),
                ("classes_from", self.classes_from),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "give exactly one of `colormap` (masks stored as pictures), `labels` (masks "
                "stored as label maps, which is how volumes do it), `regions` "
                "(overlapping classes as unions of a label map's values) and `classes_from` "
                "(one binary mask file per class, for classes that overlap independently)"
                + (f"; got {given}" if given else "; got none of them")
            )
        return self

    @property
    def n_class(self) -> int:
        """How many classes there are. One of the three namings decides; nothing else.

        With `regions` this is the number of regions and there is no background among them,
        so the same count means something different: `labels: [0, 1, 2, 4]` is four channels
        one of which is background, and three regions over the same file is three channels
        none of which is. `classes_from` counts its patterns and has no background either,
        for the same reason - which is why `_class_fingerprint` records *which* naming a run
        used rather than only how many classes it had.
        """
        for naming in (self.colormap, self.labels, self.regions, self.classes_from):
            if naming is not None:
                return len(naming)
        raise AssertionError("one_way_of_naming_classes should have refused this spec")

    @property
    def class_names(self) -> list[str]:
        """One SHORT label per class, in channel order. A name where there is one.

        Not the same job as `EngineBase._describe_classes`, which writes a sentence for a
        refusal - *"region 'absent' (of [9])"* - and reads well in a message and badly as a
        column heading. This is the label a caller pairs with a file on disk or a row in a
        table, and the two were conflated once: `maju116/rplatypus#259` asked R to name a
        class the way the engine does, and the engine's way turned five of R's own
        assertions from `"one, both"` into `"region 'one' (of [1]), region 'both' (of
        [1, 2])"`.

        A region and a `classes_from` pattern have names of their own, so those are used.
        `labels` and `colormap` do not: there the number **is** the class, and inventing
        "class 2" as a label would read as a name somebody chose. The index is the honest
        answer and is what R has always shown for them.

        Here rather than in R because R held a copy that covered `regions` alone, numbered
        the other three, and claimed in its own documentation to mirror the engine. One
        definition, and §3A's rule decides where it lives: the engine's name wins.
        """
        if self.regions is not None:
            return [str(region.name) for region in self.regions]
        if self.classes_from is not None:
            return [str(pattern) for pattern in self.classes_from]
        return [str(index) for index in range(self.n_class)]

    @property
    def label_map(self) -> bool:
        """Whether masks are label maps rather than pictures."""
        return self.labels is not None or self.regions is not None

    @property
    def multilabel(self) -> bool:
        """Whether a voxel may belong to more than one class.

        The single switch the rest of the package reads. It decides a sigmoid per channel
        instead of a softmax across them, a per-channel threshold instead of an `argmax`,
        and whether `save_masks` can write a colour map at all.

        **Two namings answer yes** and everything downstream reads this rather than either
        of them: `regions` derives overlapping channels from one label map's values, and
        `classes_from` reads one binary file per class. From the target onwards the two are
        identical - which is what made the second one cheap, and is the whole reason
        `regions` was built first.
        """
        return self.regions is not None or self.classes_from is not None

    @property
    def multilabel_naming(self) -> str:
        """Which field made the channels independent, for a message that has to name it.

        A refusal that says `regions` to somebody who wrote `classes_from` sends them
        looking for a field they have not got.
        """
        return "regions" if self.regions is not None else "classes_from"

    def measurement_fingerprint(self) -> dict:
        """The base's `window`, plus the two fields only this task has.

        Both default to `None`, so a run that named neither says so by their absence rather
        than by two `null`s - the author's decision of 2026-11-08, and the reason the base's
        docstring gives for it. That docstring also carries the measurement for each of the
        three and the question this deliberately leaves open.
        """
        recorded = super().measurement_fingerprint()
        if self.channels_from is not None:
            recorded["channels_from"] = list(self.channels_from)
        if self.target_spacing is not None:
            recorded["target_spacing"] = list(self.target_spacing)
        return recorded
