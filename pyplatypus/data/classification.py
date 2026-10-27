"""One image, one label, and the two places that label can be written.

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
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
    """

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

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
            **{f"n_{key}": float(value) for key, value in self.counts.items()},
        }

    def __str__(self) -> str:
        rows = "\n".join(
            f"  {name:<24s} {count:>7d}  {count / self.total:6.2%}"
            for name, count in self.counts.items()
        )
        name, _ = self.rarest
        return (
            f"{self.total} images in {len(self.counts)} classes\n{rows}\n"
            f"  rarest: {name} at {self.prevalence:.2%} - below about 10% a metric read off "
            f"the hard prediction stops telling a model from a collapse"
        )


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


def discover_labelled(
    root: str | Path, data: ClassificationData, *, only_images: bool = False, strict: bool = True
) -> Discovery:
    """The samples under `root`, each carrying the name of its class.

    **A class named in `classes` and holding no image is refused**, by name. The
    alternative is to train on what is there, and then nothing says which class the model
    cannot possibly have learnt: every per-class metric would refuse at the end of the
    first epoch instead, three layers from the cause, and accuracy would report a figure
    that looks fine. `classes` is the thing to shorten when a class genuinely is not in a
    dataset, which is a sentence somebody writes rather than a directory somebody forgot.
    """
    root = Path(root)
    if not root.exists():
        raise ConfigError(f"'{root}' does not exist")

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
        )
        if only_images:
            return found
        samples, skipped = _check_labels(
            list(found.samples), data.classes, data.strict_labels, root
        )
        skipped = list(found.skipped) + skipped

    if not only_images:
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

    def labels(self) -> np.ndarray:
        """Every sample's class index, in order - what a stratified split would read."""
        return np.array([self.label_of(sample) for sample in self.samples], dtype=np.int64)

    def balance(self) -> ClassBalance:
        """Counted in `classes` order rather than in whatever order the labels arrived."""
        seen = Counter(sample.label for sample in self.samples)
        return ClassBalance(counts={name: seen.get(name, 0) for name in self.data.classes})

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64 | None]:
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
        return image, np.int64(self.label_of(sample))
