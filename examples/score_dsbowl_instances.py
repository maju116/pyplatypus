"""Score the published `dsbowl-unet` weights as instances, which is what the challenge scored.

    python examples/score_dsbowl_instances.py

The model card for those weights says it plainly: **semantic, not instance.** Touching nuclei
come back as one region, so its Dice is not comparable to the 2018 Data Science Bowl's own
leaderboard, which scored instances. That caveat has been true and unmeasured since the
weights were published, and this script is what measures it.

**It builds no model and trains nothing.** It loads the published weights by name, turns their
semantic masks into instances the naive way - connected components - and scores them the way
the challenge did. The number it produces is the baseline every model in this milestone gets
measured against, which is why nothing here is tuned: no minimum area, no morphology, no
threshold sweep. A baseline that has been tuned is no longer the thing later work is compared
against.

What it reads is already on disk in the right form. The Data Science Bowl stores one PNG per
nucleus - 70 files in one validation case, 375 in another, 2 in the smallest - and
`unite_masks()` is what the training pipeline does with them: the element-wise maximum, which
throws instance identity away at read time. Nothing new has to be collected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import pyplatypus as pp
from pyplatypus.instances import (
    instances_from_mask,
    read_instances,
    score_cases,
    summarise_instances,
)

WEIGHTS = "dsbowl-unet"


def specification(train: str, validation: str, size: int) -> dict:
    """The published weights, over the split they were measured on, trained not at all.

    The same two paths and the same `config_file` mode `publish_dsbowl_weights.py` used, for
    one reason: the split has to be the card's split or the two numbers cannot be put side by
    side. `split_dataset` with the card's fractions and seed is what reproduces it, and the
    Dice agreeing with the sidecar is what proves it did.
    """
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": train,
            "validation_path": validation,
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "mode": "config_file",
        },
        "models": [
            {
                "name": "nuclei",
                "architecture": "u_net",
                "input_shape": [size, size],
                "channels": 3,
                "blocks": 4,
                "filters": 16,
                "batch_size": 4,
                "epochs": 1,
                "loss": {"name": "cce_dice"},
                "metrics": [{"name": "dice", "include_background": False}],
                "weights": WEIGHTS,
                "fit": False,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="examples/data/data_science_bowl/stage1_train",
        help="a directory of case directories - all of them, which the split then divides",
    )
    parser.add_argument("--size", type=int, default=256)
    # The card's own split, so the instance score and the published Dice are the same cases.
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--published-dice", type=float, default=0.9205236150257623)
    parser.add_argument(
        "--splits",
        default=None,
        help="where to write the split CSVs; beside the data by default, never inside it",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="where to write the rows as JSON")
    parser.add_argument("--worst", type=int, default=10)
    arguments = parser.parse_args()

    data = Path(arguments.data)
    cases_on_disk = sorted(path for path in data.iterdir() if path.is_dir())
    print(f"{len(cases_on_disk)} case directories under {data}")

    # Never inside `data`: a directory there is discovered as a case, so the split would
    # become a sample of itself on the second run and the counts would move with no reason
    # visible in the output.
    splits = Path(arguments.splits) if arguments.splits else data.parent / "instance-splits"
    if splits.resolve() == data.resolve() or data.resolve() in splits.resolve().parents:
        raise SystemExit(
            f"{splits} is inside {data}, where a directory is read as a case. "
            "Point --splits somewhere else"
        )
    split = pp.split_dataset(data, splits, fractions=(0.8, 0.2), seed=arguments.seed)
    print("split:", json.dumps(split["samples"]))

    spec = pp.from_dict(
        specification(split["paths"]["train"], split["paths"]["validation"], arguments.size)
    )
    engine = pp.Engine(spec, device=arguments.device, num_workers=arguments.workers)
    engine.fit(verbose=False)
    print(f"loaded {WEIGHTS}, trained nothing")

    # `space="source"` and not the model grid: the per-object files are on the grid of the
    # image they were drawn on, and scoring a 256x256 prediction against a 360x360 truth would
    # either refuse or, worse, resize one of them and compare a different picture.
    predictions = engine.predict("nuclei", split="validation", space="source")
    samples = engine.dataset(spec.models[0], "validation").samples
    print(f"predicted {len(predictions)} cases on their own grids")

    cases = []
    for sample, probabilities in zip(samples, predictions, strict=True):
        mask = np.argmax(probabilities, axis=-1)
        cases.append((sample.key, read_instances(data / sample.key), instances_from_mask(mask)))
    if not cases:
        raise SystemExit("nothing was predicted; the validation split came back empty")

    rows = score_cases(cases)
    summary = summarise_instances(rows)

    # The pair is the point, so both halves are measured here rather than one being quoted
    # from the model card. The card's Dice was measured on 134 held-out images and this runs
    # on 134 case directories, and those are **not** automatically the same 134 - a mismatch
    # of exactly that shape was found on this dataset once and only naming the cases showed
    # it. So the Dice below comes from the same engine, the same weights and the same rows,
    # joined on the case name, and the join is asserted rather than assumed.
    dice = {row["case"]: row["dice"] for row in engine.evaluate_cases("nuclei", "validation")}
    missing = [row["case"] for row in rows if row["case"] not in dice]
    if missing:
        raise SystemExit(
            f"{len(missing)} cases were scored for instances but not for Dice, "
            f"starting with {missing[0]}; the two halves are not the same cases"
        )
    semantic = float(np.mean([dice[row["case"]] for row in rows]))
    summary["mean_dice_same_cases"] = semantic

    # The control, and the reason the split above is reproduced rather than assumed. If this
    # Dice is not the sidecar's, the cases are not the sidecar's cases and the instance score
    # below cannot be set beside the published number - which is a mistake this dataset has
    # already caused once: the 134 directories sitting in `stage1_validation` on one machine
    # are a different 134 from the ones the card measured, and the card's worst case is not
    # among them.
    if abs(semantic - arguments.published_dice) > 1e-6:
        raise SystemExit(
            f"Dice came out {semantic:.7f} and the sidecar published "
            f"{arguments.published_dice:.7f}. Same weights, so these are not the same cases - "
            "do not read the instance score beside the published Dice until that is understood"
        )
    print(f"control: Dice reproduces the sidecar's {arguments.published_dice:.7f} exactly")

    print("\n--- the baseline, scored the way the challenge scored it ---")
    for key, value in summary.items():
        print(f"  {key:26} {value if not isinstance(value, float) else round(value, 4)}")

    print(f"\n--- the {arguments.worst} worst cases ---")
    for row in sorted(rows, key=lambda row: row["average_precision"] or 0.0)[: arguments.worst]:
        matched = next(value for key, value in row.items() if key.startswith("matched_at_"))
        print(
            f"  {row['case'][:12]}  ap {row['average_precision']:.4f}  "
            f"truth {row['truth']:4}  predicted {row['predicted']:4}  matched {matched:4}"
        )

    print(
        f"\n--- what the pair says ---\n"
        f"  Dice over the same {len(rows)} cases, on the model's grid   {semantic:.4f}\n"
        f"  instance average precision, on the source grid        {summary['mean_average_precision']:.4f}\n"
        "  The card for these weights says *semantic, not instance*. That is the size of it."
    )

    found = sum(row["predicted"] for row in rows)
    truth = sum(row["truth"] for row in rows)
    print(
        f"\nconnected components found {found} objects where the data has {truth}: "
        f"{found / truth:.1%}. Every nucleus a merged region swallows is both a miss and, "
        "through the region that swallowed it, a false positive."
    )

    if arguments.out:
        Path(arguments.out).write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
