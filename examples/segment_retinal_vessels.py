"""Retinal vessels on FIVES: the first example that tiles, and the first judged on clDice.

Three things here appear in no other example, and each is the reason this dataset was
chosen rather than a canonical one.

**It tiles.** A FIVES fundus is 2048 x 2048 and its vessels are about nine pixels wide -
measured, area over skeleton length. Resized to 256 they are 1.2 pixels, and at one pixel a
displacement of one pixel leaves no overlap at all, so no overlap-based metric can tell a
near miss from a total one. `splits` reads the source at full resolution and cuts it into
tiles the model's size, which is what the field is for and what nothing else demonstrates.

**It asks for a metric a tile cannot answer.** A skeleton is a property of a whole mask, and
a tile severs every vessel crossing its edge, so a perfectly connected model would score low
and say nothing. `evaluate_cases` therefore holds a case's tiles until its image is complete
and measures `cldice` once on the reassembled pair - which is why it sits in `metrics` beside
`dice` here rather than in a second pass. It is still absent from the epoch, where a
per-batch average would be the meaningless version, so `early_stopping` watches `val_dice`.

**The scores are reported per disease.** FIVES is 200 cases each of normal, AMD, diabetic
retinopathy and glaucoma, and the disease is in the file name. A model that works on healthy
retinas and fails on diabetic retinopathy is the interesting finding, and one mean hides it.

What this run actually found is a warning about reading the result too quickly. Glaucoma came
back with several times the spread of the other three, twice, on independent runs - and it is
not the disease. FIVES ships a `Quality Assessment.xlsx` grading every image for illumination,
blur and contrast, and glaucoma is where the poor grades are: on the images it grades clean,
glaucoma is the *best* of the four and the tightest. Before concluding that a group is hard,
check whether it is the group whose pictures are worse.

    python examples/segment_retinal_vessels.py --data examples/data/fives

The data is one 1.7 GB download from figshare 19688169, CC BY 4.0, and the authors' own
train/test split is already balanced across the four groups - 150 and 50 of each - so it is
used rather than replaced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pyplatypus as pp

FOLDER = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
GROUPS = {"N": "normal", "A": "AMD", "D": "diabetic retinopathy", "G": "glaucoma"}
BY_DISEASE = r"_([NADG])$"


def write_splits(root: Path, out_dir: Path) -> dict[str, Path]:
    """One CSV per split, each row naming an image, its mask and **a key**.

    The key is the file's own name. Without it a `config_file` sample is called `row 55`,
    which names nothing on disk and cannot be grouped by - and grouping is most of what this
    example has to say.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ("train", "test"):
        images = sorted((root / FOLDER / split / "Original").glob("*.png"))
        if not images:
            raise SystemExit(f"no images under {root / FOLDER / split}; see the module docstring")
        rows = ["key,images,masks"]
        for image in images:
            mask = root / FOLDER / split / "Ground truth" / image.name
            if not mask.exists():
                raise SystemExit(f"{image.name} has no mask beside it")
            rows.append(f"{image.stem},{image.resolve()},{mask.resolve()}")
        path = out_dir / f"{split}.csv"
        path.write_text("\n".join(rows) + "\n")
        written[split] = path
        print(f"  {split}: {len(images)} cases -> {path}")
    return written


def specification(splits: dict[str, Path], arguments) -> dict:
    """The run, as the dict a YAML file would have parsed into.

    `cldice` sits among the metrics beside `dice`, which is the part that was impossible
    until 0.8.0a6. A skeleton is a property of a whole mask and a tile has none, so the
    combination used to be refused and this example carried forty lines of second pass to
    work around it. `evaluate_cases` now holds a case's tiles until its image is complete
    and measures it once on the reassembled pair, so asking is a line of configuration.

    It is still absent from the epoch: a per-batch average of a per-tile skeleton would be
    meaningless, so `val_cldice` does not exist on a tiled run and `early_stopping` watches
    `val_dice`. The specification refuses the other pairing when it is read.
    """
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(splits["train"]),
            "validation_path": str(splits["test"]),
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "mode": "config_file",
        },
        "models": [
            {
                "name": "vessels",
                "architecture": "u_net",
                "input_shape": [arguments.size, arguments.size],
                # 2048 read whole and cut into a grid of `size` tiles. Training shuffles in
                # windows so one decode serves all of a sample's tiles; without that this
                # example costs twice as long an epoch.
                "splits": [arguments.tiles, arguments.tiles],
                "blocks": 4,
                "filters": arguments.filters,
                "batch_size": arguments.batch,
                "epochs": arguments.epochs,
                "loss": {"name": arguments.loss},
                "metrics": [
                    {"name": "dice", "include_background": False},
                    {"name": "iou", "include_background": False},
                    # Measured on the reassembled image, per case, by `evaluate_cases`.
                    {
                        "name": "cldice",
                        "iterations": arguments.iterations,
                        "include_background": False,
                    },
                ],
                "optimizer": {"name": "adam", "learning_rate": arguments.rate},
                "callbacks": [
                    {
                        "name": "early_stopping",
                        "monitor": "val_dice",
                        "patience": arguments.patience,
                        "restore_best": True,
                    }
                ],
                "augmentation": [
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "VerticalFlip", "params": {"p": 0.5}},
                ],
            }
        ],
        "seed": arguments.seed,
    }


def by_group(rows: list[dict], column: str) -> dict[str, dict]:
    """Mean and spread per disease, because one number over four diseases hides the one
    that matters."""
    import re

    out: dict[str, list[float]] = {}
    for row in rows:
        found = re.search(BY_DISEASE, row["case"])
        out.setdefault(GROUPS[found.group(1)] if found else "?", []).append(row[column])
    return {
        name: {
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
        }
        for name, values in sorted(out.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="examples/data/fives")
    parser.add_argument("--size", type=int, default=512, help="the tile, divisible by 32")
    parser.add_argument("--tiles", type=int, default=4, help="2048 / size")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--loss", default="cce_dice")
    parser.add_argument(
        "--iterations",
        type=int,
        default=12,
        help="skeleton peels; a FIVES vessel is about nine pixels across",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="measurements/fives")
    parser.add_argument("--figures", default=None, help="a directory to draw the worst cases into")
    arguments = parser.parse_args()

    root = Path(arguments.data)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = write_splits(root, out / "splits")

    config = specification(splits, arguments)
    spec = pp.from_dict(config)
    engine = pp.build_engine(spec, num_workers=arguments.workers)

    started = time.time()
    history = engine.fit(verbose=True)["vessels"]
    print(f"\ntrained in {(time.time() - started) / 60:.1f} min, {len(history.records)} epochs")

    print("\n--- the split as one number ---")
    for row in engine.evaluate():
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    # One pass. dice and iou come off the tiles' overlap counts; cldice is measured on the
    # reassembled image, inside the same walk over the split.
    started = time.time()
    cases = engine.evaluate_cases("vessels")
    print(f"\nscored {len(cases)} cases in {(time.time() - started) / 60:.1f} min")

    for column in ("dice", "cldice"):
        print(f"\n--- {column} per disease ---")
        for name, stats in by_group(cases, column).items():
            print(
                f"  {name:22s} n={stats['n']:3d}  mean {stats['mean']:.4f}  "
                f"sd {stats['sd']:.4f}  worst {stats['min']:.4f}"
            )

    gaps = [(row["case"], row["dice"], row["cldice"]) for row in cases]
    gaps.sort(key=lambda item: item[2] - item[1])
    print("\n  where the two disagree most (case, dice, clDice):")
    for case, dice, cld in gaps[:3] + gaps[-3:]:
        print(f"    {case:12s} {dice:.4f}  {cld:.4f}   {cld - dice:+.4f}")

    (out / "fives.json").write_text(
        json.dumps(
            {
                "configuration": config,
                "history": history.records,
                "dice_by_disease": by_group(cases, "dice"),
                "cldice_by_disease": by_group(cases, "cldice"),
                "cases": cases,
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(f"\nwrote {out / 'fives.json'}")

    if arguments.figures:
        figures = Path(arguments.figures)
        figures.mkdir(parents=True, exist_ok=True)
        # Whole images, not tiles. `predict` reassembles, so a tile from the dataset would
        # be a 512-pixel truth against a 2048-pixel prediction - which `plot_masks` refuses,
        # and which is how this was caught rather than shipped.
        dataset = engine.dataset(engine.runs["vessels"].spec, "validation")
        worst = min(range(len(cases)), key=lambda i: cases[i]["dice"])
        which = [0, worst]

        def whole_image(sample):
            array = pp.read_image(sample.images[0], channels=3, size=None)
            return np.asarray(array, dtype=np.float32) / 255.0

        def whole_mask(sample):
            mask = pp.read_image(sample.masks[0], channels=1, size=None, nearest=True)
            flat = (np.asarray(mask).reshape(mask.shape[:2]) > 127).astype(np.float32)
            return np.stack([1 - flat, flat], axis=-1)

        # Two images out of two hundred, so the stream is read for the two it is asked
        # for and the rest are dropped as they arrive. `predict` would hand back all 200
        # at full resolution to draw two of them.
        wanted = {cases[i]["case"]: i for i in which}
        drawn = {
            case: whole
            for case, whole in engine.predict_stream("vessels", split="validation")
            if case in wanted
        }
        figure = pp.plot_masks(
            np.stack([whole_image(dataset.samples[i]) for i in which]),
            prediction=np.stack([drawn[cases[i]["case"]] for i in which]),
            truth=np.stack([whole_mask(dataset.samples[i]) for i in which]),
            labels=[f"{cases[i]['case']}  dice {cases[i]['dice']:.3f}" for i in which],
        )
        figure.savefig(figures / "fives-masks.png", dpi=110, bbox_inches="tight")
        print(f"wrote {figures / 'fives-masks.png'}")


if __name__ == "__main__":
    main()
