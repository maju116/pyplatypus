"""What `target_spacing` buys, and what it costs when the field of view is wrong.

    python examples/measure_target_spacing.py --data examples/data/msd-spleen/Task09_Spleen

`target_spacing` is documented with an argument and no measurement. This gives it one, and
the first thing it measures is not the scores: it is the dataset.

## Two readings taken before anything was trained

**The physical scale really does vary here**, which is what makes the dataset worth using:

    slice thickness     1.5 mm to 8.0 mm        9 distinct values over 41 volumes
    in plane            0.613 to 0.977 mm       39 distinct values
    extent along z      192.5 mm to 520 mm      2.70x
    voxel volume        0.853 to 5.030 mm3      5.90x

**And `target_spacing` is not a free switch, which is the finding this example exists for.**
It converts `input_shape` from *the whole image, squashed* into *this many millimetres,
centred* - and the organ has to be inside that box. Measured over all 41 labels:

    spleen extent, at most              117 x 133 x 143 mm
    spleen centre off the volume's, at most  127 x  81 x 165 mm
    a field of view that always contains it  371 x 296 x 472 mm
      which at 1.5 mm is                     248 x 198 x 315 voxels = 15.5 M

So the obvious comparison - `target_spacing: [1.5, 1.5, 1.5]` against a plain resize, same
`input_shape` - cannot be run at a resolution where the resampled arm sees the spleen at all.
Run anyway it measures cropping, not resampling.

## Three arms, so the trap and the effect are separate numbers

    resize      no target_spacing         the whole volume squashed into input_shape
    naive       target_spacing 1.5 mm     96 mm centred: what a reader of the docs would try
    covering    target_spacing 7.5 mm     480 mm centred: a field of view that contains it

Same `input_shape`, same model, same epochs. The middle arm is the one a reader reaches for
and it is here to be measured rather than warned about.

## Scored on the grid the files are stored on

Every arm is read back with `predict(space="source")` and scored against the truth as
stored, so three models at three physical scales are compared on one grid. Comparing them at
their own `load_shape` would be the rig `maju116/rplatypus#187` is about - and these three
share a `load_shape`, which is why that warning does not fire and is not enough here.

The volume error in millilitres is beside the Dice, because that is the quantity a
radiologist writes down and it is what the spacing was carried for.

## The data

MSD Task09 (Spleen), **CC BY-SA 4.0** - from `license.txt` in the dataset's own store and
`dataset.json` inside the archive. No weights are published here; the artefact is the table.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import pyplatypus as pp
from pyplatypus.data.volumes import read_volume, volume_spacing

ARMS = {
    "resize": None,
    "naive": [1.5, 1.5, 1.5],
    "covering": [7.5, 7.5, 7.5],
}


def write_splits(root: Path, out: Path, seed: int) -> dict[str, Path]:
    images = sorted((root / "imagesTr").glob("spleen_*.nii.gz"))
    rows = [
        (p.name.replace(".nii.gz", ""), p.resolve(), (root / "labelsTr" / p.name).resolve())
        for p in images
        if (root / "labelsTr" / p.name).exists()
    ]
    random.Random(seed).shuffle(rows)
    cuts = (round(0.6 * len(rows)), round(0.8 * len(rows)))
    parts = {
        "train": rows[: cuts[0]],
        "validation": rows[cuts[0] : cuts[1]],
        "test": rows[cuts[1] :],
    }
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, part in parts.items():
        path = out / f"{name}.csv"
        path.write_text("key,images,masks\n" + "".join(f"{k},{i},{m}\n" for k, i, m in part))
        written[name] = path
    print(f"  {len(rows)} volumes: " + ", ".join(f"{n} {len(p)}" for n, p in parts.items()))
    return written


def specification(splits: dict[str, Path], spacing, size: int, epochs: int) -> dict:
    data = {
        "train_path": str(splits["train"]),
        "validation_path": str(splits["validation"]),
        "test_path": str(splits["test"]),
        "mode": "config_file",
        "labels": [0, 1],
        "window": "soft_tissue",
    }
    if spacing is not None:
        data["target_spacing"] = spacing
    return {
        "task": "semantic_segmentation",
        "data": data,
        "models": [
            {
                "name": "spleen",
                "architecture": "u_net",
                "input_shape": [size, size, size],
                "channels": 1,
                "blocks": 3,
                "filters": 16,
                "batch_size": 1,
                "epochs": epochs,
                "loss": {"name": "cce_dice"},
                "metrics": [{"name": "dice", "include_background": False}],
                "optimizer": {"name": "adam", "learning_rate": 0.001},
                # One field, and `#204` is why: without a checkpoint a trained model cannot
                # be asked anything once the process exits, so a reading nobody thought of
                # in advance costs the whole run again.
                "callbacks": [
                    {
                        "name": "model_checkpoint",
                        "monitor": "val_dice",
                        "path": "measurements/spleen/checkpoints/{model}.safetensors",
                    }
                ],
            }
        ],
        "seed": 1,
    }


def score_at_source(engine, splits: dict[str, Path]) -> dict:
    """Dice and millilitres against the truth as stored, for every test volume.

    Not `evaluate_cases`, which scores each model at its own `load_shape` - and here all
    three arms share one, so it would look comparable and would not be: a voxel is 1.5 mm
    in one arm and 7.5 in another.
    """
    rows = [r.split(",") for r in splits["test"].read_text().strip().split("\n")[1:]]
    predicted = engine.predict("spleen", split="test", space="source")
    cases = []
    for (key, _image, mask), answer in zip(rows, predicted, strict=True):
        truth = read_volume(mask, nearest=True)[..., 0] > 0.5
        found = answer.argmax(axis=-1) == 1
        total = float(truth.sum() + found.sum())
        spacing = volume_spacing(mask)
        voxel_ml = float(np.prod(spacing)) / 1000
        cases.append(
            {
                "case": key,
                "dice": 2 * float((truth & found).sum()) / total if total else 1.0,
                "true_ml": float(truth.sum()) * voxel_ml,
                "predicted_ml": float(found.sum()) * voxel_ml,
                # Per case, because the whole argument is that these differ between scans.
                # Without them a worst case is a number with no story, and `#204` paid for
                # that once: a run keeping only aggregates cannot be asked which case
                # failed, and the model is gone by the time anyone wants to know.
                "spacing": [float(s) for s in spacing],
                "extent_mm": [float(s * n) for s, n in zip(spacing, truth.shape, strict=True)],
            }
        )
    dice = [c["dice"] for c in cases]
    error_ml = [c["predicted_ml"] - c["true_ml"] for c in cases]
    return {
        "dice": float(np.mean(dice)),
        "worst_dice": float(np.min(dice)),
        "worst_case": min(cases, key=lambda c: c["dice"])["case"],
        "mean_absolute_error_ml": float(np.mean(np.abs(error_ml))),
        "mean_signed_error_ml": float(np.mean(error_ml)),
        "mean_true_ml": float(np.mean([c["true_ml"] for c in cases])),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/msd-spleen/Task09_Spleen")
    parser.add_argument("--out", default="measurements/spleen")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()

    out = Path(arguments.out)
    print("--- the split, by volume because one volume is one patient here ---")
    splits = write_splits(Path(arguments.data), out / "spacing-split", seed=1)

    results = {}
    for name, spacing in ARMS.items():
        field = (
            "the whole volume" if spacing is None else f"{spacing[0] * arguments.size:.0f} mm cubed"
        )
        print(f"\n--- {name}: {field} ---")
        config = specification(splits, spacing, arguments.size, arguments.epochs)
        engine = pp.build_engine(pp.from_dict(config), num_workers=arguments.workers)
        engine.fit(verbose=False)
        results[name] = score_at_source(engine, splits)
        row = results[name]
        print(
            f"  dice {row['dice']:.4f}  worst {row['worst_dice']:.4f}  "
            f"|error| {row['mean_absolute_error_ml']:.1f} ml of {row['mean_true_ml']:.1f} ml"
        )

    print("\n--- per case, with the geometry that is supposed to explain it ---")
    reference = results["covering"]["cases"]
    header = f"{'case':14s}{'z extent':>10s}{'voxel':>9s}" + "".join(f"{n:>10s}" for n in ARMS)
    print("  " + header)
    for index, case in enumerate(reference):
        scores = "".join(f"{results[n]['cases'][index]['dice']:10.4f}" for n in ARMS)
        extent = case["extent_mm"][2]
        print(f"  {case['case']:14s}{extent:8.0f}mm{np.prod(case['spacing']):9.2f}{scores}")

    print("\n--- all three on the grid the files are stored on ---")
    for name in ARMS:
        row = results[name]
        print(
            f"  {name:9s} dice {row['dice']:.4f}  worst {row['worst_dice']:.4f}  "
            f"|error| {row['mean_absolute_error_ml']:6.1f} ml  "
            f"signed {row['mean_signed_error_ml']:+7.1f} ml"
        )

    (out / "target-spacing.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n  wrote {out / 'target-spacing.json'}")


if __name__ == "__main__":
    main()
