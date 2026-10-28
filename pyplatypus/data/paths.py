"""Finding the images and the masks.

Two layouts, both inherited from the old package because they are what the examples use:

* `nested_dirs` - one directory per sample, holding an `images/` and a `masks/`
  subdirectory. The Data Science Bowl puts 27 separate mask files in one sample, one per
  nucleus, so masks are always a list.
* `config_file` - a CSV with `images` and `masks` columns, several paths per cell
  separated by `column_sep`.

One deliberate change: the old version caught `FileNotFoundError`, logged a warning and
carried on, so a sample missing its masks silently vanished from training. Warnings in a
loop over 536 directories are warnings nobody reads. Here an incomplete sample is an
error by default, and skipping is something the caller asks for.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pyplatypus.errors import ConfigError
from pyplatypus.spec.common import DataMode
from pyplatypus.spec.data import DataSpec


@dataclass(frozen=True)
class Sample:
    """One image, and whatever labels it - files for a mask or a name for a label.

    `label` is the classification task's answer and is a **name rather than a path**, which
    is why it is a field here instead of a one-entry `masks`: everything downstream that
    touches `masks` treats its entries as files to read, and `("benign",)` would reach a
    reader as a path that does not exist. It is `None` for every other task, so one
    `Discovery` serves all of them and `split_samples` and `write_splits` need no branch.

    `labels` is the same answer when the label is a **set** - the classes a chest film's
    four findings say are present - and it is a second field rather than a tuple in the
    first one. The reason is that nothing downstream would notice: `stratified_split`
    strata by `sample.label`, so a tuple there would become a stratum per *combination*
    without one word being changed, and `_check_labels` compares it against `classes` with
    an `in` that a tuple satisfies for the wrong reason. Two fields make every reader say
    which it is asking for, and exactly one of them is set.

    **The empty tuple is a real answer and `None` is not one**, which is why the default is
    `None`: a film with no findings is a labelled image whose set is empty, and a reader
    that could not tell that from "this task does not have sets" would train on it as
    unlabelled.
    """

    key: str
    images: tuple[Path, ...]
    masks: tuple[Path, ...] = ()
    label: str | None = None
    labels: tuple[str, ...] | None = None

    @property
    def image(self) -> Path:
        return self.images[0]


@dataclass(frozen=True)
class Discovery:
    samples: tuple[Sample, ...]
    skipped: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.samples)


def _nested_dirs(
    root: Path, subdirs: tuple[str, str], only_images: bool
) -> tuple[list[Sample], list[tuple[str, str]]]:
    samples: list[Sample] = []
    skipped: list[tuple[str, str]] = []

    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = entry / subdirs[0]
        if not image_dir.is_dir():
            skipped.append((entry.name, f"no '{subdirs[0]}' directory"))
            continue
        images = tuple(sorted(p for p in image_dir.iterdir() if p.is_file()))
        if not images:
            skipped.append((entry.name, f"'{subdirs[0]}' is empty"))
            continue

        masks: tuple[Path, ...] = ()
        if not only_images:
            mask_dir = entry / subdirs[1]
            if not mask_dir.is_dir():
                skipped.append((entry.name, f"no '{subdirs[1]}' directory"))
                continue
            masks = tuple(sorted(p for p in mask_dir.iterdir() if p.is_file()))
            if not masks:
                skipped.append((entry.name, f"'{subdirs[1]}' is empty"))
                continue

        samples.append(Sample(key=entry.name, images=images, masks=masks))
    return samples, skipped


#: What a cell may say when a column is one class's yes-or-no. Written forms of the two
#: integers only, with `0.0` and `1.0` because a CSV written by pandas writes floats and a
#: reader that refused those would be refusing a file nobody typed by hand.
_BINARY_CELLS = {"0": False, "1": True, "0.0": False, "1.0": True}


def _binary_cell(cell: str, column: str, key: str, path: Path) -> bool:
    """One column's answer for one image, or a refusal naming the cell.

    **Refused rather than coerced, and the case this is written for is `-1`.** CheXpert and
    the datasets shaped like it write `-1` for *uncertain* and leave the cell *empty* for
    not mentioned, and both have a defensible mapping to 0 or to 1 - which is the point:
    the mapping is a decision about somebody's data and about what their paper claims, so
    the reader that guesses it has made that decision silently and reported nothing. The
    refusal says so and names the two forms it takes, because the fix is one pass over the
    CSV rather than anything in a specification.
    """
    text = cell.strip()
    if text in _BINARY_CELLS:
        return _BINARY_CELLS[text]
    if not text:
        raise ConfigError(
            f"'{key}' in '{path}' leaves column '{column}' empty, and an empty cell is not "
            f"a 0: in the datasets written this way it means the finding was not mentioned, "
            f"which is a different claim from absent. Decide which it is for your data and "
            f"write {' or '.join(sorted(_BINARY_CELLS))} in the cell."
        )
    raise ConfigError(
        f"'{key}' in '{path}' has '{text}' in column '{column}', and a column naming one "
        f"class holds that class's yes-or-no: {', '.join(sorted(_BINARY_CELLS))}. '-1' is "
        f"the uncertain label some datasets write and mapping it to either answer is a "
        f"decision about your data rather than one this reader can take."
    )


def _config_file(
    path: Path,
    column_sep: str,
    only_images: bool,
    label_column: str = "masks",
    *,
    label_is_a_name: bool = False,
    label_columns: Sequence[str] | None = None,
) -> tuple[list[Sample], list[tuple[str, str]]]:
    """One CSV reader for every task, because the half that differs is one cell.

    `label_is_a_name` is the classification path: its column holds a class name rather than
    a file, so the cell is kept as text instead of being resolved against the CSV's
    directory. A second reader would have been the alternative and it would have had to
    repeat the two things this one gets right - that a relative image path resolves against
    the CSV rather than the working directory, and that a `key` column beats a row number -
    which is the duplication §3F of the workbench's notes is a list of.

    `label_columns` is the same path one step further: a label that is a set is **one
    column per class** rather than one column holding a name, which is `classes_from`'s
    shape - one binary file per class - written into a CSV. It takes precedence over
    `label_column`, so a file carrying both is read as a set and its `label` column is one
    of the many columns this reader ignores.

    **The order of `label_columns` is the class order and this reader does not know it**,
    which is why it is a sequence rather than a set: the caller read it off `classes`, and
    the tuple this writes comes back out in the same order so that nothing downstream has
    to re-derive it.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ConfigError(f"'{path}' contains no rows")
    if "images" not in rows[0]:
        found = ", ".join(rows[0].keys())
        raise ConfigError(f"'{path}' needs an 'images' column; found: {found}")
    if not only_images and label_columns is not None:
        missing = [name for name in label_columns if name not in rows[0]]
        if missing:
            found = ", ".join(rows[0].keys())
            raise ConfigError(
                f"'{path}' needs one column per class and is missing "
                f"{', '.join(missing)}; found: {found}"
            )
    elif not only_images and label_column not in rows[0]:
        found = ", ".join(rows[0].keys())
        raise ConfigError(f"'{path}' needs a '{label_column}' column; found: {found}")

    # Relative paths resolve against the CSV, not the working directory, so a config
    # file travels with its data instead of only working from one place.
    base = path.parent

    def resolve(cell: str) -> tuple[Path, ...]:
        out = []
        for piece in cell.split(column_sep):
            piece = piece.strip()
            if piece:
                candidate = Path(piece)
                out.append(candidate if candidate.is_absolute() else base / candidate)
        return tuple(out)

    samples: list[Sample] = []
    skipped: list[tuple[str, str]] = []
    for number, row in enumerate(rows, start=2):  # row 1 is the header
        images = resolve(row.get("images") or "")
        label = None
        labels: tuple[str, ...] | None = None
        masks: tuple[Path, ...] = ()
        # A `key` column wins over the row number, because the row number names nothing.
        # `write_splits` puts the original sample name there, so a case that scores badly
        # can be found on disk instead of being reported as 'row 55'. Read before the
        # labels rather than after them, because a refusal about a cell has to be able to
        # say which row it was in.
        key = (row.get("key") or "").strip() or f"row {number}"
        if not only_images:
            if label_columns is not None:
                labels = tuple(
                    name
                    for name in label_columns
                    if _binary_cell(row.get(name) or "", name, key, path)
                )
            elif label_is_a_name:
                label = (row.get(label_column) or "").strip() or None
            else:
                masks = resolve(row.get(label_column) or "")
        if not images:
            skipped.append((key, "no image path"))
            continue
        # `labels` is excluded from this test on purpose: an empty tuple is an image with
        # none of the findings, which is a labelled image and the commonest row in every
        # dataset this reads. A row whose cells could not be read never gets here - it was
        # refused above rather than skipped, because an unreadable cell is not a missing
        # label and recording it as one would hide it behind `strict: false`.
        if not only_images and labels is None and not masks and label is None:
            skipped.append((key, f"no {label_column} " + ("name" if label_is_a_name else "path")))
            continue
        samples.append(Sample(key=key, images=images, masks=masks, label=label, labels=labels))
    return samples, skipped


def finish(
    root: Path, samples: list[Sample], skipped: list[tuple[str, str]], strict: bool
) -> Discovery:
    """The tail every walk shares: refuse an empty result, and refuse an incomplete one.

    Extracted when the classification walk arrived, because that walk is genuinely
    different - one directory per class rather than one per sample - while what to do about
    what it could not use is not.
    """
    if not samples:
        raise ConfigError(
            f"'{root}' yielded no usable samples"
            + (f"; {len(skipped)} were incomplete" if skipped else "")
        )
    if skipped and strict:
        shown = "\n".join(f"  - {key}: {why}" for key, why in skipped[:10])
        more = f"\n  ... and {len(skipped) - 10} more" if len(skipped) > 10 else ""
        raise ConfigError(
            f"{len(skipped)} of {len(samples) + len(skipped)} samples under '{root}' are "
            f"incomplete:\n{shown}{more}\n"
            "Pass strict=False to train on the rest instead."
        )
    return Discovery(samples=tuple(samples), skipped=tuple(skipped))


def discover_samples(
    root: str | Path,
    *,
    mode: DataMode = DataMode.NESTED_DIRS,
    subdirs: tuple[str, str] = ("images", "masks"),
    column_sep: str = ";",
    only_images: bool = False,
    strict: bool = True,
    label_column: str = "masks",
    label_is_a_name: bool = False,
    label_columns: Sequence[str] | None = None,
) -> Discovery:
    """List the samples under `root`, given only the layout.

    Separate from `discover` because finding files does not need a whole specification:
    splitting a dataset has no model, no colormap and no loss, and should not have to
    invent them.
    """
    root = Path(root)
    if not root.exists():
        raise ConfigError(f"'{root}' does not exist")

    if mode is DataMode.NESTED_DIRS:
        if not root.is_dir():
            raise ConfigError(f"mode is nested_dirs but '{root}' is not a directory")
        samples, skipped = _nested_dirs(root, subdirs, only_images)
    else:
        if not root.is_file():
            raise ConfigError(f"mode is config_file but '{root}' is not a file")
        samples, skipped = _config_file(
            root,
            column_sep,
            only_images,
            label_column,
            label_is_a_name=label_is_a_name,
            label_columns=label_columns,
        )

    return finish(root, samples, skipped, strict)


def discover(
    root: str | Path, data: DataSpec, *, only_images: bool = False, strict: bool = True
) -> Discovery:
    """List the samples under `root`, which is one of the paths named in `data`.

    `DataSpec` rather than `SegmentationData`: finding files needs the layout and nothing
    else, so this already works for a detection spec, whose second subdirectory holds
    annotation files instead of masks.

    **It needs a sample to be a directory of two things, which is one layout and not the
    only one.** A classification spec leaves `subdirs` unset on purpose - its sample is a
    single image and its class is the directory that image sits in - so this refuses by
    name rather than handing `None` to the walk and failing several frames down on
    something that reads like a corrupt dataset.
    """
    if data.subdirs is None:
        raise ConfigError(
            f"{type(data).__name__} names no `subdirs`, and this reads the layout where a "
            f"sample is a directory holding the image in one subdirectory and its labels "
            f"in the other. A classification dataset is one directory per class, which is "
            f"a different walk and has its own reader."
        )
    return discover_samples(
        root,
        mode=data.mode,
        subdirs=data.subdirs,
        column_sep=data.column_sep,
        only_images=only_images,
        strict=strict,
        label_column=data.label_column,
    )
