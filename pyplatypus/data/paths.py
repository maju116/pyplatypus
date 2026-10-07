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
from dataclasses import dataclass
from pathlib import Path

from pyplatypus.errors import ConfigError
from pyplatypus.spec.common import DataMode
from pyplatypus.spec.data import DataSpec


@dataclass(frozen=True)
class Sample:
    """One image and the mask files that belong to it."""

    key: str
    images: tuple[Path, ...]
    masks: tuple[Path, ...] = ()

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


def _config_file(
    path: Path, column_sep: str, only_images: bool, label_column: str = "masks"
) -> tuple[list[Sample], list[tuple[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ConfigError(f"'{path}' contains no rows")
    if "images" not in rows[0]:
        found = ", ".join(rows[0].keys())
        raise ConfigError(f"'{path}' needs an 'images' column; found: {found}")
    if not only_images and label_column not in rows[0]:
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
        masks = () if only_images else resolve(row.get(label_column) or "")
        # A `key` column wins over the row number, because the row number names nothing.
        # `write_splits` puts the original sample name there, so a case that scores badly
        # can be found on disk instead of being reported as 'row 55'.
        key = (row.get("key") or "").strip() or f"row {number}"
        if not images:
            skipped.append((key, "no image path"))
            continue
        if not only_images and not masks:
            skipped.append((key, f"no {label_column} path"))
            continue
        samples.append(Sample(key=key, images=images, masks=masks))
    return samples, skipped


def discover_samples(
    root: str | Path,
    *,
    mode: DataMode = DataMode.NESTED_DIRS,
    subdirs: tuple[str, str] = ("images", "masks"),
    column_sep: str = ";",
    only_images: bool = False,
    strict: bool = True,
    label_column: str = "masks",
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
        samples, skipped = _config_file(root, column_sep, only_images, label_column)

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


def discover(
    root: str | Path, data: DataSpec, *, only_images: bool = False, strict: bool = True
) -> Discovery:
    """List the samples under `root`, which is one of the paths named in `data`.

    `DataSpec` rather than `SegmentationData`: finding files needs the layout and nothing
    else, so this already works for a detection spec, whose second subdirectory holds
    annotation files instead of masks.
    """
    return discover_samples(
        root,
        mode=data.mode,
        subdirs=data.subdirs,
        column_sep=data.column_sep,
        only_images=only_images,
        strict=strict,
        label_column=data.label_column,
    )
