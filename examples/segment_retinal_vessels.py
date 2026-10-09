"""Retinal vessels on FIVES: the first example that tiles, and the first judged on clDice.

Three things here appear in no other example, and each is the reason this dataset was
chosen rather than a canonical one.

**It tiles.** A FIVES fundus is 2048 x 2048 and its vessels are about nine pixels wide -
measured, area over skeleton length. Resized to 256 they are 1.2 pixels, and at one pixel a
displacement of one pixel leaves no overlap at all, so no overlap-based metric can tell a
near miss from a total one. `splits` reads the source at full resolution and cuts it into
tiles the model's size, which is what the field is for and what nothing else demonstrates.

**It is scored twice, on purpose.** Dice and IoU come from the tiled run. clDice does not and
cannot: a skeleton is a property of a whole mask, and a tile severs every vessel crossing its
edge, so a perfectly connected model would score low and say nothing. The specification
refuses the combination. So the second pass steps the model over one retina's tiles, stitches
them, and scores that - see `cldice_per_case` for why it cannot use `predict`.

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
import torch

import pyplatypus as pp
from pyplatypus.data.images import stitch
from pyplatypus.objectives.functional import cldice_from_masks

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

    No `cldice` among the metrics, and that is not an oversight: the specification refuses a
    whole-mask metric on a tiled run, because a tile has no skeleton worth the name. It is
    measured in a second pass below.
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


def predict_one(engine, dataset, index: int) -> np.ndarray:
    """One sample's prediction, tiles stitched, channels-last.

    `engine.predict` does a whole split at once and holds three copies of it on the way -
    every tile's probabilities, their concatenation, and the stack of stitched images. At
    200 retinas of 2048 that is tens of gigabytes and it killed a run that had finished
    training. Nothing here needs the whole split in memory, so nothing here asks for it.
    """
    run = engine.runs["vessels"]
    model, spec, device = run.trainer.model, run.spec, run.trainer.device
    tiles = dataset.tiles_per_sample
    batch = max(1, spec.batch_size)

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, tiles, batch):
            stack = torch.stack(
                [
                    torch.as_tensor(dataset[index * tiles + offset][0]).permute(2, 0, 1)
                    for offset in range(start, min(start + batch, tiles))
                ]
            ).to(device)
            out = model(stack)
            if isinstance(out, tuple):
                out = out[-1]
            pieces.append(np.moveaxis(out.softmax(dim=1).cpu().numpy(), 1, -1))

    probabilities = np.concatenate(pieces, axis=0)
    if spec.splits is None:
        return probabilities[0]
    return stitch(probabilities, tuple(spec.splits))


def cldice_per_case(engine, split: str, iterations: int) -> list[dict]:
    """clDice on whole images, one image at a time.

    Whole images because a skeleton is a property of a whole mask: `predict` reassembles the
    tiles, so this scores the picture the model would actually hand back rather than sixteen
    pieces of it.

    One at a time because `predict` cannot do this split at all. It holds every tile's
    probabilities, concatenates them into a second copy and stacks the stitched images into
    a third - for 200 retinas at 2048 that is tens of gigabytes, and a run that had finished
    training was killed here. So the model is stepped over one sample's tiles, stitched,
    scored, and the arrays dropped: peak memory is one image whatever the split's size.

    The masks are read by the same reader the dataset uses, at the same size, because a mask
    read another way is a mask of something else.
    """
    run = engine.runs["vessels"]
    dataset = engine.dataset(run.spec, split)
    # On whichever device trained, because the skeleton is twelve rounds of pooling over a
    # four-megapixel tensor: measured at 11.96 s an image on the processor and 0.37 on the
    # card, which is forty minutes against ninety seconds over this split.
    device = run.trainer.device

    rows = []
    for index, sample in enumerate(dataset.samples):
        whole = predict_one(engine, dataset, index)

        hard = torch.as_tensor(whole, device=device).permute(2, 0, 1)[None]
        hard = (
            torch.nn.functional.one_hot(hard.argmax(dim=1), hard.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )
        mask = pp.read_image(sample.masks[0], channels=1, size=None, nearest=True)
        flat = (np.asarray(mask).reshape(mask.shape[:2]) > 127).astype(np.float32)
        target = torch.as_tensor(np.stack([1 - flat, flat]), device=device)[None]
        if target.shape[-2:] != hard.shape[-2:]:
            raise SystemExit(
                f"{sample.key}: prediction is {tuple(hard.shape[-2:])} and the mask is "
                f"{tuple(target.shape[-2:])}; clDice on two different grids is a number "
                "about nothing"
            )
        score = cldice_from_masks(hard, target, iterations, smooth=0.0)[0, 1].item()
        rows.append({"case": sample.key, "cldice": score})
    return rows


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
    parser.add_argument("--figures", default=None)
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

    cases = engine.evaluate_cases("vessels")
    print("\n--- dice per disease, from the tiled run ---")
    for name, stats in by_group(cases, "dice").items():
        print(
            f"  {name:22s} n={stats['n']:3d}  mean {stats['mean']:.4f}  "
            f"sd {stats['sd']:.4f}  worst {stats['min']:.4f}"
        )

    # clDice cannot be read off a tiled run, so it gets a pass of its own over whole images.
    print("\n--- clDice, on reassembled whole images ---")
    started = time.time()
    topology = cldice_per_case(engine, "validation", arguments.iterations)
    print(f"  {len(topology)} images in {(time.time() - started) / 60:.1f} min")
    for name, stats in by_group(topology, "cldice").items():
        print(
            f"  {name:22s} n={stats['n']:3d}  mean {stats['mean']:.4f}  "
            f"sd {stats['sd']:.4f}  worst {stats['min']:.4f}"
        )

    paired = {row["case"]: row["dice"] for row in cases}
    gaps = [
        (row["case"], paired[row["case"]], row["cldice"])
        for row in topology
        if row["case"] in paired
    ]
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
                "cldice_by_disease": by_group(topology, "cldice"),
                "cases": cases,
                "cldice": topology,
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

        figure = pp.plot_masks(
            np.stack([whole_image(dataset.samples[i]) for i in which]),
            prediction=np.stack([predict_one(engine, dataset, i) for i in which]),
            truth=np.stack([whole_mask(dataset.samples[i]) for i in which]),
            labels=[f"{cases[i]['case']}  dice {cases[i]['dice']:.3f}" for i in which],
        )
        figure.savefig(figures / "fives-masks.png", dpi=110, bbox_inches="tight")
        print(f"wrote {figures / 'fives-masks.png'}")


if __name__ == "__main__":
    main()
