"""Splitting a dataset into train, validation and test - by patient, not by slice.

The single most common way to get a good-looking segmentation result that means nothing:
put slices from one patient into training and validation at the same time. The model sees
the neighbouring slice of the same anatomy in training, so validation measures memory
rather than generalisation, and Dice comes out several points too high with nothing in the
output to say so. Radiology datasets invite it, because they arrive as one flat pile of
slices whose filenames are the only clue about who they came from.

So splitting here is group-aware from the start. `group_by` is a regular expression read
against the sample key; everything sharing a group lands in exactly one split. A group is
usually a patient, and may be a study, a scanner or a site - whatever the unit is that
must not straddle the split.

Two deliberate refusals:

* a `group_by` that matches nothing is an error, never a fallback to one-group-per-sample.
  A silent fallback is exactly the inflated-score bug, reintroduced by a typo.
* the result is written as three CSV files listing paths, not as three copies of the data.
  Medical datasets are large and often read-only, and a split that copies gigabytes is a
  split nobody runs twice.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

from pyplatypus.data.paths import Sample, discover_samples
from pyplatypus.errors import ConfigError
from pyplatypus.spec.common import DataMode

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Split:
    """Which samples went where, and which group each one belongs to."""

    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]
    groups: Mapping[str, str]

    def __getitem__(self, name: str) -> tuple[Sample, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"no split called '{name}'; there are: {', '.join(SPLIT_NAMES)}")
        return getattr(self, name)

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(self[name]) for name in SPLIT_NAMES}

    @property
    def group_counts(self) -> dict[str, int]:
        return {name: len({self.groups[s.key] for s in self[name]}) for name in SPLIT_NAMES}

    def groups_of(self, name: str) -> set[str]:
        return {self.groups[s.key] for s in self[name]}


def group_of(key: str, pattern: str | None) -> str:
    """The group a sample belongs to, from its key.

    Without a pattern every sample is its own group, which is the right answer when one
    sample is one patient. With one, the first capture group is the name if the expression
    has one, otherwise the whole match.
    """
    if pattern is None:
        return key
    found = re.search(pattern, key)
    if found is None:
        raise ConfigError(
            f"group_by '{pattern}' does not match the sample key '{key}'. Every sample "
            "must belong to a group; a pattern that matches nothing would quietly make "
            "each sample its own patient, which is the mistake this argument exists to "
            "prevent."
        )
    return found.group(1) if found.groups() else found.group(0)


def _fractions(fractions: Sequence[float] | Mapping[str, float]) -> dict[str, float]:
    if isinstance(fractions, Mapping):
        unknown = set(fractions) - set(SPLIT_NAMES)
        if unknown:
            raise ConfigError(
                f"unknown split(s) {', '.join(sorted(unknown))}; "
                f"there are: {', '.join(SPLIT_NAMES)}"
            )
        shares = {name: float(fractions.get(name, 0.0)) for name in SPLIT_NAMES}
    else:
        if len(fractions) not in (2, 3):
            raise ConfigError(
                f"give two or three fractions (train, validation[, test]), got {len(fractions)}"
            )
        values = [float(v) for v in fractions] + [0.0]
        shares = dict(zip(SPLIT_NAMES, values[:3], strict=True))

    if any(v < 0 for v in shares.values()):
        raise ConfigError(f"fractions cannot be negative: {shares}")
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"fractions must add up to 1, these add up to {total:g}: {shares}")
    if shares["train"] <= 0 or shares["validation"] <= 0:
        raise ConfigError(
            "train and validation must both get a positive share - a model with no "
            "validation set cannot be told whether it learned anything"
        )
    return shares


def split_samples(
    samples: Sequence[Sample],
    *,
    fractions: Sequence[float] | Mapping[str, float] = (0.7, 0.15, 0.15),
    group_by: str | None = None,
    seed: int = 0,
) -> Split:
    r"""Divide samples between the splits, keeping every group whole.

    Deterministic: the same samples, fractions, pattern and seed give the same split on
    any machine. That is not a nicety - a split that moves between runs makes two results
    incomparable, and the reason for a difference becomes impossible to find.

    Groups are shuffled, then handed out one at a time to whichever split is furthest
    below its target share *in samples*. Assigning by group count instead would drift
    badly whenever groups differ in size, which for patients they always do.

    Args:
        samples: What was discovered. Nothing is read from disk here - this partitions a
            list, so a specification with a `split` block gains no side effect on disk.
        fractions: Two or three, summing to one: train, validation, and optionally test.
            A mapping names them instead, for a split that is not in that order.
        group_by: A regular expression read against each sample's key; everything sharing
            a group lands in exactly one split. `None` means every sample is its own
            group, which is right only when the samples are genuinely independent - slices
            of one patient are not.
        seed: Which shuffle. The split is otherwise deterministic.

    Returns:
        A `Split`, with `train`, `validation` and `test` tuples and the group each sample
        was assigned to.

    Raises:
        SplitError: If the fractions are not two or three, if `group_by` matches no sample
                key - a pattern that matches nothing is a mistake, not an empty result - or if
                there are fewer groups than splits to fill.

    >>> from pyplatypus.data.paths import Sample
    >>> import pathlib
    >>> samples = [
    ...     Sample(key=f"patient{p:02d}_slice{s}", images=(pathlib.Path("x.png"),))
    ...     for p in range(1, 5)
    ...     for s in range(3)
    ... ]
    >>> len(samples)
    12
    >>> split = split_samples(samples, fractions=(0.5, 0.5), group_by=r"^(patient\d+)_")
    >>> len(split.train) + len(split.validation)
    12

    A patient is never on both sides, which is the whole point:

    >>> def patients(part):
    ...     return {key.split("_")[0] for key in (sample.key for sample in part)}
    >>> patients(split.train) & patients(split.validation)
    set()

    Without `group_by`, every slice is its own group and the same patient lands in both -
    which makes validation measure memory rather than generalisation:

    >>> loose = split_samples(samples, fractions=(0.5, 0.5))
    >>> bool(patients(loose.train) & patients(loose.validation))
    True
    """
    if not samples:
        raise ConfigError("there are no samples to split")

    shares = _fractions(fractions)
    groups: dict[str, list[Sample]] = {}
    membership: dict[str, str] = {}
    for sample in samples:
        name = group_of(sample.key, group_by)
        groups.setdefault(name, []).append(sample)
        membership[sample.key] = name

    wanted = [name for name in SPLIT_NAMES if shares[name] > 0]
    if len(groups) < len(wanted):
        raise ConfigError(
            f"{len(groups)} group(s) cannot fill {len(wanted)} splits "
            f"({', '.join(wanted)}). With one group per patient, a split needs at least "
            "as many patients as it has parts - or pass a coarser group_by, or ask for "
            "fewer splits."
        )

    order = sorted(groups)
    Random(seed).shuffle(order)

    total = len(samples)
    targets = {name: shares[name] * total for name in wanted}
    assigned: dict[str, str] = {}
    sizes = dict.fromkeys(wanted, 0)

    for name in order:
        # Largest deficit first; ties break on the declared order, so train wins over
        # validation and validation over test. Deterministic either way.
        chosen = max(wanted, key=lambda s: (targets[s] - sizes[s], -wanted.index(s)))
        assigned[name] = chosen
        sizes[chosen] += len(groups[name])

    _fill_empty_splits(assigned, groups, wanted)

    held: dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    for name in order:
        held[assigned[name]].extend(groups[name])

    return Split(
        train=tuple(held["train"]),
        validation=tuple(held["validation"]),
        test=tuple(held["test"]),
        groups=membership,
    )


def _fill_empty_splits(
    assigned: dict[str, str], groups: Mapping[str, list[Sample]], wanted: Sequence[str]
) -> None:
    """Make sure every requested split got something, by moving one group if it did not.

    Chasing the target *sizes* can leave a split empty when the groups are lopsided: one
    patient with 100 slices and two with one each fills train and starves test, however the
    shuffle came out. An empty validation or test set is useless, and refusing the whole
    split over an unlucky shuffle is worse than being slightly off the requested fractions,
    so the smallest group is moved out of whichever split holds the most.
    """
    for name in wanted:
        if any(target == name for target in assigned.values()):
            continue
        counts: dict[str, list[str]] = {}
        for group, target in assigned.items():
            counts.setdefault(target, []).append(group)
        donor = max(counts, key=lambda s: (len(counts[s]), -wanted.index(s)))
        if len(counts[donor]) < 2:
            raise ConfigError(
                f"the '{name}' split came out empty and no other split has a group to "
                "spare. Ask for fewer splits, or use a coarser group_by."
            )
        moved = min(counts[donor], key=lambda g: (len(groups[g]), g))
        assigned[moved] = name


def write_splits(
    split: Split,
    out_dir: str | Path,
    *,
    column_sep: str = ";",
    relative: bool = True,
    label_column: str = "masks",
) -> dict[str, Path]:
    """Write one CSV per split, in the layout `config_file` mode reads.

    Paths are written relative to the CSV when they can be, so the dataset and its split
    can be moved or mounted elsewhere together. `key` and `group` columns come along for
    provenance: the reader ignores them, a person reading the file does not, and they are
    what lets anyone check afterwards that no patient straddles the split.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        samples = split[name]
        if not samples:
            continue
        path = out / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["key", "group", "images", label_column])
            for sample in samples:
                writer.writerow(
                    [
                        sample.key,
                        split.groups[sample.key],
                        column_sep.join(_as_text(p, out, relative) for p in sample.images),
                        column_sep.join(_as_text(p, out, relative) for p in sample.masks),
                    ]
                )
        written[name] = path
    return written


def _as_text(path: Path, base: Path, relative: bool) -> str:
    if not relative:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        # Not under the output directory. An absolute path still works from anywhere,
        # which is better than a fragile chain of '..' segments.
        return str(path.resolve())


def split_dataset(
    root: str | Path,
    out_dir: str | Path,
    *,
    mode: DataMode | str = DataMode.NESTED_DIRS,
    subdirs: tuple[str, str] = ("images", "masks"),
    column_sep: str = ";",
    fractions: Sequence[float] | Mapping[str, float] = (0.7, 0.15, 0.15),
    group_by: str | None = None,
    seed: int = 0,
    strict: bool = True,
    relative: bool = True,
    label_column: str = "masks",
) -> dict[str, object]:
    r"""Split one folder of data into three CSVs a specification can point at.

    The whole point of the function: a researcher has one directory and needs
    `train_path`, `validation_path` and `test_path`. Returns the paths it wrote plus the
    counts, so the caller can see what happened without opening the files.

    `label_column` names the second column, and it has to match what the specification
    will look for - `masks` for segmentation, `annotations` for detection. Not derived
    from `subdirs`, although that would read well: someone whose directories are called
    `img` and `lbl` has always got a `masks` column out of this, and deriving it would
    quietly write a file their existing configuration could no longer read.

    The other way to divide one folder is the specification's own `split` block, which
    partitions the samples in memory and writes nothing. Use this one when the CSVs are
    the point - to keep, to hand to a colleague, or to cite in a paper.

    Args:
        root: The folder holding the data.
        out_dir: Where the three CSVs go.
        mode: `nested_dirs` for one directory per sample, `config_file` for a CSV.
        subdirs: For `nested_dirs`, the image and label subdirectories.
        column_sep: What separates several paths **inside one cell** - a sample with one
            mask file per object, or one file per channel. Not the CSV's own delimiter,
            which is a comma: that is the distinction this argument's name does not make.
        fractions: Two or three, as for `split_samples`.
        group_by: A pattern picking the group out of each sample's key - a patient,
            usually. As for `split_samples`, and for the same reason.
        seed: Which shuffle.
        strict: Whether a sample that cannot be read is an error rather than a skip.
        relative: Write paths relative to `out_dir`, so the CSVs survive being moved
            together with the data.
        label_column: The name of the second column.

    Returns:
        `paths` - the CSV written for each split that got any; `samples` and `groups` -
        how many of each went where, test included and zero when there are two fractions;
        and `skipped`, which is not zero when `strict` is False and something could not be
        read.

    >>> import pathlib, tempfile
    >>> root = pathlib.Path(tempfile.mkdtemp())
    >>> for patient in range(1, 5):
    ...     for slice_no in range(2):
    ...         case = root / f"patient{patient:02d}_slice{slice_no}"
    ...         (case / "images").mkdir(parents=True)
    ...         (case / "masks").mkdir(parents=True)
    ...         _ = (case / "images" / "scan.png").write_bytes(b"")
    ...         _ = (case / "masks" / "mask.png").write_bytes(b"")
    >>> out = split_dataset(
    ...     root,
    ...     root / "splits",
    ...     fractions=(0.5, 0.5),
    ...     group_by=r"^(patient\d+)_",
    ... )
    >>> sorted(out)
    ['groups', 'paths', 'samples', 'skipped']
    >>> out["samples"]
    {'train': 4, 'validation': 4, 'test': 0}

    Two groups each side, because a patient is never divided:

    >>> out["groups"]
    {'train': 2, 'validation': 2, 'test': 0}

    And the CSVs are what a specification points at. Note the header: the file is
    comma-separated and carries the key and the group it was assigned, while `column_sep`
    separates several paths within a single cell.

    >>> written = pathlib.Path(out["paths"]["train"]).read_text().splitlines()
    >>> written[0]
    'key,group,images,masks'
    >>> len(written) - 1
    4
    """
    found = discover_samples(
        root,
        mode=DataMode(mode),
        subdirs=subdirs,
        column_sep=column_sep,
        strict=strict,
        label_column=label_column,
    )
    split = split_samples(found.samples, fractions=fractions, group_by=group_by, seed=seed)
    paths = write_splits(
        split, out_dir, column_sep=column_sep, relative=relative, label_column=label_column
    )
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "samples": split.counts,
        "groups": split.group_counts,
        "skipped": len(found.skipped),
    }
