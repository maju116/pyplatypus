"""Train the separation model #159 proposed, and score it as instances.

    python examples/train_instance_separation.py --data path/to/stage1_train --width 3

**The answer was no, and this script is the evidence rather than a recommendation.** #159 is
closed measured-and-no: a boundary separation signal is worth +0.0640 of instance average
precision on this data at its best, and its best absolute score is 0.5587 where a perfect
semantic mask with no separation at all reaches 0.7655. Anyone reaching for this as a way to
do instance segmentation should read that table first - it is below.

It is kept because a negative result nobody can reproduce is an opinion. Every row of the
table came out of this file with `--width`, and the diagnostics it prints are what made the
result readable rather than merely numeric.

    rim   fg Dice  interior  rim Dice  objects  matched@0.5   AP sep   AP cc   effect
    1 px   0.9188    0.9012    0.4808     4923         4020   0.5169  0.4893  +0.0276
    2 px   0.9199    0.8908    0.6419     5086         4248   0.5486  0.4907  +0.0579
    3 px   0.9186    0.8554    0.7292     5481         4513   0.5587  0.4947  +0.0640
    4 px   0.9171    0.8068    0.7622     5525         4438   0.5306  0.4791  +0.0515

5774 objects in the truth; two runs of one configuration differ by 0.0074, which is the noise
every gap above is quoted against. The reason the curve has a maximum is in the two middle
columns: **a rim wide enough to learn is wide enough to hollow the object out**, so rim Dice
climbs while interior Dice falls, and past 3 px the interiors are too small to say where an
object ends. Past the peak the model even finds *more* regions and matches *fewer* of them.

Three mutually exclusive classes - background, object interior, object rim - which is why
this needs nothing new from the engine: it is `labels: [0, 1, 2]` on the softmax path, the
same U-Net the published `dsbowl-unet` weights are, with one more class.

**Since `#160` there is a shorter way to say the same thing**: `task: instance_segmentation`
and `separation: {name: rim}` builds that target from the per-object files rather than asking
for a three-class label map to be written to disk first, and
`examples/compare_instance_separations.py` is that path. This script stays as the record of
the table above - `derive_parts` wrote the maps the four runs were measured on, and a result
whose code was rewritten underneath it is not reproducible.

**The comparison is deliberately one variable.** Same architecture, same blocks and filters,
same input size, same loss, same split, same seed, same epochs as the card for those weights
records. The only difference is that the mask names three classes instead of two. Anything
else moved and the result would say "something changed", which this dataset has already shown
it cannot distinguish: its four architectures land inside 0.0030 of each other while the
spread across images within any one of them is 0.054.

What it reports, and all three are needed to read the result:

    foreground Dice      is the mask as good as the two-class model's 0.9205? If this
                         falls, the instance number moved for a reason that is not separation
    instance AP          the thing #159 is about
    the two ceilings      0.7655 for connected components and 0.9846 for this post-process,
                         both measured on the ground truth over the same 134 cases

**The bar is 0.7655, not 0.4888.** Connected components caps any model at 0.7655 on this data
because touching nuclei are one region, so beating the published weights' 0.4888 proves
nothing about separation; passing 0.7655 proves something connected components cannot do.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import pyplatypus as pp
from pyplatypus.instances import (
    instances_by_watershed,
    instances_from_mask,
    instances_from_parts,
    parts_from_instances,
    read_instances,
    score_cases,
    summarise_instances,
)

#: What the card for `dsbowl-unet` records, so the only thing that differs is the class count.
PUBLISHED = {"blocks": 4, "filters": 16, "input_shape": [256, 256], "loss": "cce_dice"}


def derive_parts(source: Path, target: Path, width: int) -> int:
    """Write a three-class label map per case, with the image symlinked beside it.

    The images are linked rather than copied: 670 of them, and a copy is a second thing that
    can drift from the first. The masks are written, because they do not exist yet - the Data
    Science Bowl ships one binary file per nucleus and the decomposition is what this script
    is for.
    """
    written = 0
    for case in sorted(path for path in source.iterdir() if path.is_dir()):
        out = target / case.name
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        for image in sorted((case / "images").iterdir()):
            link = out / "images" / image.name
            if not link.exists():
                os.symlink(image.resolve(), link)
        destination = out / "masks" / "parts.png"
        if not destination.exists():
            interior, rim = parts_from_instances(read_instances(case), width=width)
            classes = np.zeros(interior.shape, np.uint8)
            classes[interior] = 1
            classes[rim] = 2
            Image.fromarray(classes).save(destination)
            written += 1
    return written


def specification(train: str, validation: str, epochs: int, seed: int) -> dict:
    return {
        "task": "semantic_segmentation",
        "seed": seed,
        "data": {
            "train_path": train,
            "validation_path": validation,
            "mode": "config_file",
            # Three classes, one integer per pixel, and the order is the contract: 1 is the
            # interior the seeds come from and 2 is the rim that keeps neighbours apart.
            "labels": [0, 1, 2],
        },
        "models": [
            {
                "name": "separation",
                "architecture": "u_net",
                "channels": 3,
                "blocks": PUBLISHED["blocks"],
                "filters": PUBLISHED["filters"],
                "input_shape": PUBLISHED["input_shape"],
                "batch_size": 32,
                "epochs": epochs,
                "loss": {"name": PUBLISHED["loss"]},
                "metrics": [{"name": "dice"}, {"name": "iou"}],
                "callbacks": [
                    {
                        "name": "early_stopping",
                        "monitor": "val_dice",
                        "patience": 10,
                        "restore_best": True,
                    },
                ],
            }
        ],
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/data_science_bowl/stage1_train")
    parser.add_argument("--derived", default=None, help="where to write the three-class masks")
    parser.add_argument("--splits", default=None)
    parser.add_argument("--epochs", type=int, default=60, help="60 is what the card records")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=1, help="pixels of rim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument(
        "--expect-cases",
        default=None,
        help="a JSON file of the run this is compared against; its case names must be the "
        "validation split exactly, which is a stronger control than any score",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse()
    data = Path(arguments.data)
    derived = Path(arguments.derived) if arguments.derived else data.parent / "parts"
    if derived.resolve() == data.resolve() or data.resolve() in derived.resolve().parents:
        raise SystemExit(f"{derived} is inside {data}, where a directory is read as a case")

    written = derive_parts(data, derived, arguments.width)
    print(
        f"three-class masks: {written} written, {len(list(derived.iterdir()))} cases under {derived}"
    )

    splits = Path(arguments.splits) if arguments.splits else derived.parent / "parts-splits"
    split = pp.split_dataset(derived, splits, fractions=(0.8, 0.2), seed=arguments.seed)
    print("split:", json.dumps(split["samples"]))

    spec = pp.from_dict(
        specification(
            split["paths"]["train"], split["paths"]["validation"], arguments.epochs, arguments.seed
        )
    )
    engine = pp.Engine(spec, device=arguments.device, num_workers=arguments.workers)
    samples = engine.dataset(spec.models[0], "validation").samples

    if arguments.expect_cases:
        expected = {
            row["case"] for row in json.loads(Path(arguments.expect_cases).read_text())["cases"]
        }
        got = {Path(sample.key).name for sample in samples}
        if got != expected:
            raise SystemExit(
                f"the validation split is {len(got)} cases and {len(got - expected)} of them are "
                "not in the run this is being compared against. Two different sets of images "
                "give numbers that look comparable and are not - do not read the score"
            )
        print(
            f"control: the validation split is exactly the {len(expected)} cases of the "
            "run this is compared against"
        )

    history = engine.fit()["separation"]
    best = history.best("val_dice") or history.records[-1]
    print(f"\ntrained {len(history)} epochs, best val_dice {best.get('val_dice'):.4f}")

    # `space="source"` and not the model grid: the per-object files are on the grid of the
    # image they were drawn on, and scoring a 256x256 prediction against a 360x360 truth
    # would either refuse or resize one of them and compare a different picture.
    predictions = engine.predict("separation", split="validation", space="source")

    separated, flooded, oracle, naive = [], [], [], []
    dice, rim_dice, interior_dice = [], [], []
    for sample, probabilities in zip(samples, predictions, strict=True):
        # The truth comes from the *original* tree, never the derived one. The derived case
        # holds a single three-class label map in `masks/`, so reading it as per-object files
        # would return two "objects" - the interior and the rim - and score that against the
        # prediction. A silent wrong answer rather than an error.
        name = Path(sample.key).name
        truth = read_instances(data / name)
        classes = probabilities.argmax(axis=-1)
        interior = classes == 1
        foreground = classes > 0
        # Needed twice below - by the oracle surface and by the per-part Dice - so computed
        # once, here, before either.
        true_interior, true_rim = parts_from_instances(truth, width=arguments.width)

        separated.append((name, truth, instances_from_parts(interior, foreground)))
        naive.append((name, truth, instances_from_mask(foreground)))
        # The same parts, flooded over the model's own rim evidence rather than split by
        # distance. One trained model, two decodings - which is why `decode` is a list.
        flooded.append(
            (name, truth, instances_by_watershed(interior, foreground, probabilities[..., 2]))
        )
        # A DIAGNOSTIC AND NOT A ROUTE: the same predicted interiors and foreground, flooded
        # over a surface built from the *ground truth* rim. It answers one question and only
        # one - is the watershed held back by the quality of the signal it climbs, or by
        # being the wrong decoding? If this row jumps, the signal is the problem; if it does
        # not move, the decoding is.
        #
        # The surface is the negated distance to the true rim rather than the rim itself:
        # a binary rim is flat, so the flood would have nothing to climb, which is exactly
        # what makes the ceiling run uninformative about this decoding.
        oracle.append(
            (
                name,
                truth,
                instances_by_watershed(
                    interior, foreground, -ndimage.distance_transform_edt(~true_rim)
                ),
            )
        )
        # The mask on its own, so a moved instance score can be attributed. Two classes
        # against three is not the comparison; foreground against foreground is.
        overlap = np.logical_and(foreground, truth > 0).sum()
        dice.append(2 * overlap / (foreground.sum() + (truth > 0).sum()))

        # The two parts on their own, because "separation did not help" has two very
        # different causes and the foreground Dice cannot tell them apart: a rim the model
        # never learned, or a rim it learned while the interiors still failed to come apart.
        for predicted, actual, into in (
            (classes == 2, true_rim, rim_dice),
            (interior, true_interior, interior_dice),
        ):
            total = predicted.sum() + actual.sum()
            into.append(
                2 * np.logical_and(predicted, actual).sum() / total if total else float("nan")
            )

    rows = score_cases(separated)
    instances = summarise_instances(rows)
    by_watershed = summarise_instances(score_cases(flooded))
    by_oracle = summarise_instances(score_cases(oracle))
    merged = summarise_instances(score_cases(naive))

    print("\n--- the mask, so a moved instance score can be attributed ---")
    print(
        f"  foreground Dice           {float(np.mean(dice)):.4f}   "
        f"(the two-class weights score 0.9205)"
    )
    print(f"  interior Dice             {float(np.nanmean(interior_dice)):.4f}")
    print(
        f"  rim Dice                  {float(np.nanmean(rim_dice)):.4f}"
        "   <- the class the whole idea rests on"
    )
    found = sum(row["predicted"] for row in rows)
    in_truth = sum(row["truth"] for row in rows)
    print(f"  objects found / in truth  {found} / {in_truth}   ({found / in_truth:.1%})")
    print("\n--- instances, three decodings of ONE trained model ---")
    print(f"  rim + nearest_seed        {instances['mean_average_precision']:.4f}")
    print(
        f"  rim + watershed           {by_watershed['mean_average_precision']:.4f}"
        "   <- surface is the predicted rim probability"
    )
    print(
        f"  connected_components      {merged['mean_average_precision']:.4f}"
        "   <- the signal thrown away entirely"
    )
    print(
        f"  rim + watershed, ORACLE   {by_oracle['mean_average_precision']:.4f}"
        "   <- surface from the TRUE rim; a diagnostic, not a route"
    )
    print("\n--- the bars, both measured on the ground truth over these cases ---")
    print("  connected components can never pass          0.7655")
    print("  this post-process can never pass             0.9846")
    print("  the published two-class weights scored       0.4888")

    print(f"\n--- the {arguments.worst} worst cases ---")
    for row in sorted(rows, key=lambda row: row["average_precision"] or 0.0)[: arguments.worst]:
        print(
            f"  {row['case'][:16]}  ap={row['average_precision']:.4f}  "
            f"truth={row['truth']:4d}  found={row['predicted']:4d}"
        )

    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(
                {
                    "epochs_run": len(history),
                    "best": best,
                    "foreground_dice": float(np.mean(dice)),
                    "interior_dice": float(np.nanmean(interior_dice)),
                    "rim_dice": float(np.nanmean(rim_dice)),
                    "instances": instances,
                    "by_watershed": by_watershed,
                    "by_watershed_oracle_surface": by_oracle,
                    "same_model_merged": merged,
                    "cases": rows,
                },
                indent=1,
            )
        )
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
