"""One image, its label, and the three places that label can be written.

The segmentation walk reads a sample as a directory holding the image in one subdirectory
and what labels it in the other. A classification sample is a single file whose class is
the directory it sits in, so this is a different walk rather than the same walk with
different arguments - which is what `discover`'s refusal has promised since the
specification landed.

**What is not different is the CSV.** `config_file` mode reads the same reader every other
task reads, with one flag saying that the label column holds a name rather than a path, for
the reason §3F of the workbench's notes gives: two readers would both have had to resolve a
relative image path against the CSV rather than the working directory, and both have had to
prefer a `key` column to a row number.

**This reader is also the only thing in the package that knows the prevalence**, which is
the number every decision about a classifier's metrics rested on. So it reports it rather
than leaving it to be discovered from a disappointing score: `ClassBalance` is what
`inspect_masks` is for a mask.

**The third place is one column per class, and it is what makes the label a SET** - a chest
film with four findings is four independent questions rather than one of four alternatives.
That is `classes_from`'s shape one task across, which is why it is a naming rather than a
switch: `colormap` and `labels` structurally cannot say that a voxel is in two classes, and
a `label` cell cannot say that a film has two findings, so a setting that only changed the
activation would train sigmoid channels against an exclusive target and report plausible
numbers meaning what softmax meant. The naming carries the claim. `pyplatypus#307`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pyplatypus.data.augmentation import Augmenter
from pyplatypus.data.images import read_image, to_float
from pyplatypus.data.paths import Discovery, Sample, discover_samples, finish
from pyplatypus.errors import ConfigError, PlatypusError
from pyplatypus.spec.classification import ClassificationData, ClassificationModel
from pyplatypus.spec.common import DataMode


class LabelError(PlatypusError):
    kind = "label_error"


@dataclass(frozen=True)
class ClassBalance:
    """How many images each class has, and how lopsided that is.

    Reported rather than computed on request, because every decision about this task's
    metrics turned on prevalence: a calibrated model never crosses the threshold at 3%, so
    accuracy, balanced accuracy and F1 all report a collapse and a real signal as the same
    number. A reader that knows the prevalence and does not say it leaves that to be
    discovered from a score that looks fine.

    **When the label is a set the counts stop being a partition**, which is the one thing
    about this class that changes: four findings on one film are four counts and one image,
    so `images` is given rather than derived and each share is that class's own. The shares
    then do not add up to 100%, and `__str__` says so where the column is - a column of
    percentages is the one thing a reader assumes sums.
    """

    counts: dict[str, int]
    images: int | None = None

    @property
    def total(self) -> int:
        """How many images were counted.

        `images` is `None` for a label that is one of a set of alternatives, where the
        counts **partition** the images and their sum is therefore the same number - and it
        is set when the label is a set, where it is not. Deriving it from the sum in both
        cases would read as a total and be the number of *labels*, which is larger than the
        dataset and divides every share by the wrong thing.
        """
        return self.images if self.images is not None else sum(self.counts.values())

    @property
    def labels_are_a_set(self) -> bool:
        """Whether an image may carry several of these classes at once.

        Read off `images` having been given rather than off the arithmetic, and the
        difference matters: a multi-label set where every image happens to carry exactly
        one finding has counts that sum to the image count, and it is still a set.
        """
        return self.images is not None

    @property
    def labels_per_image(self) -> float:
        """Labels divided by images: 1.0 exactly when the label is one of alternatives.

        The one number that says what shape the labelling is, and it is here because the
        shares cannot say it - four classes at 25% each read the same whether every image
        has one finding or a quarter of them have all four.
        """
        return sum(self.counts.values()) / self.total if self.total else 0.0

    @property
    def never(self) -> list[str]:
        """Classes no image in this set carries.

        Named because it is the reading that predicts a refusal three layers away: every
        per-class metric this task offers averages over the classes and refuses one the
        pass holds no positive of, so this is what a run is about to die of at the end of
        its first epoch.
        """
        return [name for name, count in self.counts.items() if count == 0]

    @property
    def always(self) -> list[str]:
        """Classes every image in this set carries, which only a set of labels can have.

        The other half of the same refusal: a finding present in every image has no
        negative, so a specificity and therefore a balanced accuracy have nothing to
        average. With one label out of alternatives this is empty by construction - two
        classes are named at a minimum and each image has one - which is why it costs
        nothing to ask in both cases.
        """
        total = self.total
        return [name for name, count in self.counts.items() if total and count == total]

    @property
    def rarest(self) -> tuple[str, int]:
        name = min(self.counts, key=lambda key: self.counts[key])
        return name, self.counts[name]

    @property
    def prevalence(self) -> float:
        """The rarest class's share.

        The rarest rather than the second class's, because *which* class is the interesting
        one is not this reader's business - with three classes there is no positive - and
        the number that predicts whether a threshold metric will say anything is the
        smallest share in the set.
        """
        return self.rarest[1] / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total),
            "classes": float(len(self.counts)),
            "rarest": float(self.rarest[1]),
            "prevalence": self.prevalence,
            "labels_per_image": self.labels_per_image,
            **{f"n_{key}": float(value) for key, value in self.counts.items()},
        }

    #: The share below which a metric read off the hard prediction stops distinguishing a
    #: model from a collapse. A **reference line rather than a cliff**: the measurement it
    #: comes from is at 3%, where a calibrated model never crosses the threshold and
    #: accuracy, balanced accuracy and F1 all report a real signal and a collapse as the
    #: same figure. 10% is the round number above it, and the sentence it gates says
    #: "about".
    WORTH_SAYING_BELOW = 0.10

    def __str__(self) -> str:
        rows = "\n".join(
            f"  {name:<24s} {count:>7d}  {count / self.total:6.2%}"
            for name, count in self.counts.items()
        )
        name, _ = self.rarest
        heading = f"{self.total} images in {len(self.counts)} classes"
        if self.labels_are_a_set:
            heading += f", {self.labels_per_image:.2f} labels per image"
        lines = [heading, rows]
        if self.labels_are_a_set:
            # Said where the arithmetic is visible rather than in a docstring nobody is
            # reading at that moment: the percentages in the rows above do not add up to
            # 100, and a column of shares is the one thing a reader assumes does.
            lines.append(
                "  each share is that class's own and they do not add up - a label is a set"
            )
        lines.append(f"  rarest: {name} at {self.prevalence:.2%}")
        # Before the threshold sentence, because these two are a refusal waiting to happen
        # and that one is advice.
        for named, what, why in (
            (self.never, "never present", "no positive"),
            (self.always, "present in every image", "no negative"),
        ):
            if named:
                lines.append(
                    f"  {', '.join(named)}: {what}, so every per-class metric here has "
                    f"{why} to average for it and will refuse"
                )
        # Only where it is true. A warning that fires on a balanced set is a warning people
        # learn to skip, and then it is not there on the set it was written for - which is
        # the same reasoning `terminate_on_collapse` is reported unconditionally for and
        # this one is not: that number means something at every value, and this sentence
        # does not.
        if self.prevalence < self.WORTH_SAYING_BELOW:
            lines.append(
                "  below about 10% a metric read off the hard prediction stops telling a "
                "model from a collapse, so `auc` is the one to watch"
            )
        return "\n".join(lines)


def _class_dirs(
    root: Path, classes: list[str], strict_labels: bool
) -> tuple[list[Sample], list[tuple[str, str]]]:
    """One directory per class, images inside it.

    The key is `class/filename` rather than the filename, and that is not cosmetic: two
    classes commonly hold the same file name - `0001.png` under each - and a split writes
    the key, so a duplicate key would make two different images one case.
    """
    known = set(classes)
    samples: list[Sample] = []
    skipped: list[tuple[str, str]] = []

    loose = sorted(p.name for p in root.iterdir() if p.is_file())
    if loose:
        shown = ", ".join(loose[:5]) + (" ..." if len(loose) > 5 else "")
        raise ConfigError(
            f"'{root}' holds {len(loose)} file(s) directly: {shown}. In nested_dirs mode a "
            f"classification dataset is one directory per class with images inside, so a "
            f"file at the top has no label. Move each into its class's directory, or use "
            f"config_file mode and name the label in a column."
        )

    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in entry.iterdir() if p.is_file())
        if entry.name not in known:
            if strict_labels:
                raise LabelError(
                    f"'{entry.name}' is a directory under '{root}' and is not in `classes` "
                    f"({', '.join(classes)}). With `strict_labels: true` an image whose "
                    f"label is not a named class is refused rather than dropped, because "
                    f"dropping it changes how much data there is without saying so. Add it "
                    f"to `classes`, or set `strict_labels: false` to train on the rest."
                )
            for image in images:
                skipped.append((f"{entry.name}/{image.name}", "not a named class"))
            continue
        if not images:
            skipped.append((entry.name, "no images"))
            continue
        for image in images:
            samples.append(
                Sample(key=f"{entry.name}/{image.name}", images=(image,), label=entry.name)
            )
    return samples, skipped


def _check_labels(
    samples: list[Sample], classes: list[str], strict_labels: bool, root: Path
) -> tuple[list[Sample], list[tuple[str, str]]]:
    """Labels against `classes`, for the route where the label came out of a CSV cell.

    The directory walk has already refused what it could; this is the other half, and both
    halves are needed because a CSV can name anything at all.
    """
    known = set(classes)
    kept: list[Sample] = []
    skipped: list[tuple[str, str]] = []
    for sample in samples:
        if sample.label in known:
            kept.append(sample)
            continue
        if strict_labels:
            raise LabelError(
                f"'{sample.key}' in '{root}' is labelled '{sample.label}', which is not in "
                f"`classes` ({', '.join(classes)}). With `strict_labels: true` that is "
                f"refused rather than dropped; set it to false to skip such rows instead."
            )
        skipped.append((sample.key, f"label '{sample.label}' is not a named class"))
    return kept, skipped


def _check_every_class_is_answerable(samples: list[Sample], classes: list[str], root: Path) -> None:
    """Both sides of every class, for a label that is a set.

    The single-label walk refuses a class with no images, and the argument is the same one
    step across - except that an independent question can fail in **two** directions. A
    finding no image carries has no positive to average a recall over; a finding every
    image carries has no negative to average a specificity over. `MultiLabelMetric` refuses
    both at the end of the first epoch, after the epoch is paid for, and the message it can
    write there cannot say which file to look at.

    Refused rather than reported, which is the choice `classes` makes available: a column
    that is all zeros is a class to take out of `classes` and `labels_from`, and that is a
    sentence somebody writes rather than a column somebody forgot.
    """
    # No guard for an empty `samples`, and that was a guard here for an hour: the shared
    # CSV reader calls `finish` before this is reached, and `finish` refuses a read with no
    # usable samples whatever `strict` says. So the comfortable-looking check - *say nothing
    # about the classes when nothing was read* - is unreachable, and a test written for it
    # passed with it deleted. Said here so that it is not added back on the same reasoning.
    counted = Counter(name for sample in samples for name in (sample.labels or ()))
    for named, side, what in (
        ([name for name in classes if counted[name] == 0], "no image", "a positive"),
        (
            [name for name in classes if counted[name] == len(samples)],
            "every image",
            "a negative",
        ),
    ):
        if not named:
            continue
        raise LabelError(
            f"{side} in '{root}' carries {', '.join(named)}, so there is never "
            f"{what} to score {'it' if len(named) == 1 else 'them'} against. Every metric "
            f"this task offers for a set of labels averages over the classes and refuses "
            f"one it holds a single side of - which would happen at the end of the first "
            f"epoch instead of here. Take the class out of `classes` and out of "
            f"`labels_from` if it is not really in this dataset."
        )


def discover_labelled(
    root: str | Path,
    data: ClassificationData,
    *,
    only_images: bool = False,
    strict: bool = True,
    label_columns: Sequence[str] | None = None,
) -> Discovery:
    """The samples under `root`, each carrying the name of its class - or the set of them.

    **A class named in `classes` and holding no image is refused**, by name. The
    alternative is to train on what is there, and then nothing says which class the model
    cannot possibly have learnt: every per-class metric would refuse at the end of the
    first epoch instead, three layers from the cause, and accuracy would report a figure
    that looks fine. `classes` is the thing to shorten when a class genuinely is not in a
    dataset, which is a sentence somebody writes rather than a directory somebody forgot.

    `label_columns` is one CSV column per class, in class order, and it is what makes the
    label a **set**: the sample then carries `labels` rather than `label`. It is passed in
    rather than read off `data` because the specification has no field for it yet - the
    field lands last, after the trainer, so that nothing in a configuration can describe a
    feature the run would get wrong. From slice 4 on, the engine reads it off
    `data.labels_from` and this stays the one place it is honoured.
    """
    root = Path(root)
    if not root.exists():
        raise ConfigError(f"'{root}' does not exist")

    if label_columns is not None and data.mode is DataMode.NESTED_DIRS:
        # A plain refusal rather than the message a person writing YAML needs, which is the
        # specification's to write once `labels_from` exists there: an image cannot sit in
        # two directories without being two files, so this layout cannot express a set at
        # all. Here the combination is simply impossible to serve.
        raise ConfigError(
            "a label that is a set is read from one column per class, which is "
            "`config_file` mode; `nested_dirs` makes the directory the label and an image "
            "sits in one directory"
        )
    if label_columns is not None and len(label_columns) != len(data.classes):
        raise ConfigError(
            f"one column per class, in class order: `classes` names {len(data.classes)} "
            f"and {len(label_columns)} column(s) were given. The position is the class "
            f"index, so a shorter list is not a subset - it is a different pairing."
        )

    if data.mode is DataMode.NESTED_DIRS:
        if not root.is_dir():
            raise ConfigError(f"mode is nested_dirs but '{root}' is not a directory")
        if only_images:
            # Predicting on a directory of unlabelled pictures is a different layout
            # again, and this reader is not it: the class is the directory here, so there
            # is nothing to walk without one.
            raise ConfigError(
                f"'{root}' is read as one directory per class, so there is no unlabelled "
                f"form of it. Predict from a config file naming the images, or from the "
                f"images themselves."
            )
        samples, skipped = _class_dirs(root, data.classes, data.strict_labels)
    else:
        if not root.is_file():
            raise ConfigError(f"mode is config_file but '{root}' is not a file")
        found = discover_samples(
            root,
            mode=DataMode.CONFIG_FILE,
            column_sep=data.column_sep,
            only_images=only_images,
            strict=strict,
            label_column=data.label_column,
            label_is_a_name=True,
            label_columns=label_columns,
        )
        if only_images:
            return found
        if label_columns is not None:
            # Nothing to check a name against: the columns *are* the classes, paired by
            # position, so there is no cell that could hold a class nobody named. That is
            # also why `strict_labels` has nothing to do on this route - the state it
            # chooses between cannot arise - and the specification refuses it as a written
            # field rather than this reader ignoring it quietly.
            #
            # **The column is renamed to the class here and not in the CSV reader**, which
            # knows about files and not about `classes`. The pairing is by position, so a
            # dataset whose headers are `Atelectasis` and `Effusion` reaches everything
            # downstream as whatever `classes` calls them - and a reader that left the
            # headers in would make the class index depend on the CSV, which is the one
            # thing `classes` exists to stop.
            renamed = dict(zip(label_columns, data.classes, strict=True))
            samples = [
                replace(sample, labels=tuple(renamed[name] for name in sample.labels or ()))
                for sample in found.samples
            ]
            skipped = list(found.skipped)
        else:
            samples, skipped = _check_labels(
                list(found.samples), data.classes, data.strict_labels, root
            )
            skipped = list(found.skipped) + skipped

    if not only_images and label_columns is not None:
        _check_every_class_is_answerable(samples, data.classes, root)
    elif not only_images:
        present = {sample.label for sample in samples}
        missing = [name for name in data.classes if name not in present]
        if missing:
            raise LabelError(
                f"`classes` names {', '.join(missing)} and '{root}' holds no image of "
                f"{'either' if len(missing) == 1 else 'any of them'}. A class with no "
                f"examples is a class the model cannot learn and every per-class metric "
                f"will refuse to average over - remove it from `classes` if it is not in "
                f"this dataset."
            )
    return finish(root, samples, skipped, strict)


class ClassificationDataset:
    """Samples on disk, presented as (image, class index) pairs.

    No tiling, and that is a property of the task rather than an omission: a tile of an
    image does not carry the image's label. `ClassificationModel` has no `splits` field for
    the same reason, so there is nothing here to honour.

    **The index comes from the position in `classes`**, which is what the specification
    promises its reader and the one thing in this file that must not be derived from the
    directory listing. Read off the disk, the order would be alphabetical - reproducible,
    and unrelated to anything - so a model trained where 0 meant `benign` and used where it
    means `malignant` would report plausible probabilities with nothing downstream able to
    tell.

    **Whether the label is a set is read off the samples rather than passed in**, and that
    is the same decision `DataSpec.multilabel` makes one task across: the naming carries
    the claim. A sample read from one column per class carries `labels`; one read from a
    directory or a `label` cell carries `label`. So nothing here can be told that a set of
    labels is exclusive, which is the state that trains, raises nothing and reports numbers
    meaning what softmax meant.
    """

    def __init__(
        self,
        samples: tuple[Sample, ...],
        model: ClassificationModel,
        data: ClassificationData,
        *,
        augmenter: Augmenter | None = None,
        only_images: bool = False,
    ):
        self.samples = samples
        self.model = model
        self.data = data
        self.augmenter = augmenter
        self.only_images = only_images
        self._index = {name: position for position, name in enumerate(data.classes)}
        self.multilabel = self._labels_are_a_set()

    def _labels_are_a_set(self) -> bool:
        """One shape for the whole set, or a refusal naming the samples that disagree.

        A mixed set is refused rather than resolved either way, because both resolutions
        are wrong in a way nothing downstream can see: read as exclusive, the rows carrying
        several findings lose all but one of them to the `argmax` that builds the target;
        read as a set, the rows carrying a `label` reach a sigmoid head as an all-zero
        target and the model is trained that every finding is absent from them.

        An unlabelled set - `only_images`, which is how a prediction split arrives - has
        nothing to read, and the answer there is `False` by construction rather than by
        guess: the trainer takes the activation from the specification, because a model
        that has been trained knows what it answers and the pictures it is pointed at do
        not.
        """
        kinds = {sample.labels is not None for sample in self.samples}
        if len(kinds) > 1:
            mixed = [sample.key for sample in self.samples if sample.labels is None]
            shown = ", ".join(mixed[:5]) + (" ..." if len(mixed) > 5 else "")
            raise LabelError(
                f"{len(mixed)} of {len(self.samples)} samples carry one label where the "
                f"rest carry a set of them: {shown}. One reading has to be wrong for some "
                f"of them and neither is visible in any number a run produces, so this is "
                f"refused - the labelling of a dataset is one shape."
            )
        return kinds == {True}

    #: There are no tiles, and the loader asks. One rather than a property that raises, so
    #: `make_loader`'s window logic reads the same answer it would read for an untiled
    #: segmentation dataset.
    tiles_per_sample = 1
    cache_size = 1

    def __len__(self) -> int:
        return len(self.samples)

    def label_of(self, sample: Sample) -> int:
        if sample.label is None:
            raise LabelError(f"'{sample.key}' carries no label")
        try:
            return self._index[sample.label]
        except KeyError:
            raise LabelError(
                f"'{sample.key}' is labelled '{sample.label}', which is not one of "
                f"{', '.join(self.data.classes)}"
            ) from None

    def labels_of(self, sample: Sample) -> np.ndarray:
        """One row of the multi-hot target: a column per class, 1 where it is present.

        `float32` rather than a boolean or an index, because this is the target a loss
        reads: `binary_cross_entropy` multiplies by it, and an integer target would make
        the product an integer tensor on the first term. The dtype is also what tells
        `as_onehot` to leave it alone - it passes a floating target of the right width
        through untouched, and an integer one it would expand as indices.
        """
        row = np.zeros(len(self.data.classes), dtype=np.float32)
        for name in sample.labels or ():
            try:
                row[self._index[name]] = 1.0
            except KeyError:
                raise LabelError(
                    f"'{sample.key}' carries '{name}', which is not one of "
                    f"{', '.join(self.data.classes)}"
                ) from None
        return row

    def labels(self) -> np.ndarray:
        """Every sample's label, in order: indices, or a multi-hot row each.

        **Not what the stratified split reads**, which this docstring claimed until the set
        of labels arrived: `stratified_split` works on `Sample` objects and reads
        `sample.label` itself, and a claim about a caller that does not exist is the shape
        §4 of the workbench's notes is a list of. What this is for is looking at a split's
        labelling without iterating the images, which is the only reason it is cheap.
        """
        if self.multilabel:
            return np.stack([self.labels_of(sample) for sample in self.samples])
        return np.array([self.label_of(sample) for sample in self.samples], dtype=np.int64)

    def balance(self) -> ClassBalance:
        """Counted in `classes` order rather than in whatever order the labels arrived.

        With a set of labels the image count is passed in rather than summed, because the
        counts no longer partition anything: four findings on one film are four counts and
        one image, and a total taken as their sum would be larger than the dataset.
        """
        if self.multilabel:
            seen = Counter(name for sample in self.samples for name in (sample.labels or ()))
            return ClassBalance(
                counts={name: seen.get(name, 0) for name in self.data.classes},
                images=len(self.samples),
            )
        seen = Counter(sample.label for sample in self.samples)
        return ClassBalance(counts={name: seen.get(name, 0) for name in self.data.classes})

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64 | np.ndarray | None]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is outside 0..{len(self) - 1}")

        sample = self.samples[index]
        image = read_image(
            sample.image,
            channels=self.model.channels,
            size=tuple(self.model.input_shape),
        )
        if self.augmenter is not None:
            # `mask=None`: a flip, a rotation or a crop does not move an image's label.
            # Worth one word rather than none, because a crop *can* remove the thing being
            # classified and the label will still say it is there - which is a reason to
            # choose transforms for this task carefully, not a reason for a refusal here.
            image, _ = self.augmenter(image)
        image = to_float(image)
        if self.only_images:
            return image, None
        if self.multilabel:
            return image, self.labels_of(sample)
        return image, np.int64(self.label_of(sample))
