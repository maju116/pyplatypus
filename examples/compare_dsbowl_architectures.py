"""Train all four architectures on the Data Science Bowl, in one specification.

    python examples/compare_dsbowl_architectures.py --epochs 60

One spec, four models, one comparison table - which is what this package is for, and the honest
way to decide which weights are worth publishing. Dice alone cannot answer that: a model that
scores 0.001 higher for three times the parameters and twice the epoch time is not the better
choice, it is a different trade. So the table reports the size and the cost beside the score, and
the distribution beside the mean.

Reuses the data and the split from `publish_dsbowl_weights.py`, so the numbers are comparable to
the published ones rather than to a new shuffle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyplatypus import Engine, from_dict, summarise_cases

ARCHITECTURES = ("u_net", "u_net_plus_plus", "res_u_net", "linknet")


def model(architecture: str, epochs: int, size: int, batch: int) -> dict:
    return {
        "name": architecture,
        "architecture": architecture,
        "input_shape": [size, size],
        "channels": 3,
        "n_class": 2,
        "blocks": 4,
        "filters": 16,
        "batch_size": batch,
        "epochs": epochs,
        "loss": {"name": "cce_dice"},
        "metrics": [
            {"name": "dice", "include_background": False},
            {"name": "iou", "include_background": False},
        ],
        "optimizer": {"name": "adam", "learning_rate": 0.001},
        "callbacks": [
            {"name": "reduce_lr_on_plateau", "monitor": "val_loss", "factor": 0.5, "patience": 4},
            {"name": "early_stopping", "monitor": "val_loss", "patience": 10, "restore_best": True},
        ],
        "augmentation": [
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
            {"name": "VerticalFlip", "params": {"p": 0.5}},
            {"name": "RandomRotate90", "params": {"p": 0.5}},
            {"name": "RandomBrightnessContrast", "params": {"p": 0.3}},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/bbbc038")
    parser.add_argument("--out", default="weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only", nargs="*", default=None, help="a subset of the architectures, for a smoke test"
    )
    arguments = parser.parse_args()

    splits = Path(arguments.data) / "splits"
    train, validation = splits / "train.csv", splits / "validation.csv"
    if not train.exists():
        raise SystemExit(
            f"{train} is missing. Run publish_dsbowl_weights.py first: it downloads BBBC038v1 "
            "and writes the split this script reuses, so the two sets of numbers are comparable."
        )

    wanted = arguments.only or list(ARCHITECTURES)
    spec = from_dict(
        {
            "data": {
                "train_path": str(train),
                "validation_path": str(validation),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "mode": "config_file",
            },
            "models": [
                model(name, arguments.epochs, arguments.size, arguments.batch) for name in wanted
            ],
        }
    )

    engine = Engine(spec, num_workers=arguments.workers)
    histories = engine.fit(verbose=True)

    rows = []
    for row in engine.evaluate():
        name = row["model"]
        history = histories[name]
        seconds = [record["seconds"] for record in history.records]
        cases = engine.evaluate_cases(name)
        distribution = {entry["metric"]: entry for entry in summarise_cases(cases)}
        rows.append(
            {
                "model": name,
                "parameters": row["parameters"],
                "seconds_per_epoch": sum(seconds) / len(seconds),
                "epochs_run": row["epochs_run"],
                "dice_pooled": row["dice"],
                "dice_mean": distribution["dice"]["mean"],
                "dice_sd": distribution["dice"]["sd"],
                "dice_median": distribution["dice"]["median"],
                "dice_min": distribution["dice"]["min"],
                "worst_case": distribution["dice"]["worst_case"],
                "iou_mean": distribution["iou"]["mean"],
            }
        )

    print("\n" + "=" * 100)
    print(
        f"{'model':18} {'params':>10} {'s/epoch':>8} {'epochs':>7} "
        f"{'dice mean':>10} {'sd':>7} {'median':>8} {'min':>7}"
    )
    print("-" * 100)
    for row in sorted(rows, key=lambda r: -r["dice_mean"]):
        print(
            f"{row['model']:18} {row['parameters']:>10,} {row['seconds_per_epoch']:>8.1f} "
            f"{row['epochs_run']:>7} {row['dice_mean']:>10.4f} {row['dice_sd']:>7.4f} "
            f"{row['dice_median']:>8.4f} {row['dice_min']:>7.4f}"
        )
    print("=" * 100)

    best = max(rows, key=lambda r: r["dice_mean"])
    cheapest = min(rows, key=lambda r: r["parameters"])
    spread = best["dice_mean"] - min(r["dice_mean"] for r in rows)
    print(f"\nbest by mean Dice : {best['model']} ({best['dice_mean']:.4f})")
    print(
        f"fewest parameters : {cheapest['model']} ({cheapest['parameters']:,}, "
        f"Dice {cheapest['dice_mean']:.4f})"
    )
    print(f"spread across all : {spread:.4f}")
    print(
        "\nA spread smaller than the per-image sd means the architectures are not"
        "\ndistinguishable on this data, and the cheapest one is the answer."
    )

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dsbowl-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out / 'dsbowl-comparison.json'}")

    # Every model's weights are exported. Which of them get published is a decision taken after
    # reading the table, not before: a name in the registry is a promise, and four names that mean
    # the same thing are four promises nobody needed.
    for row in rows:
        path = engine.export_weights(
            row["model"],
            out / f"dsbowl-{row['model'].replace('_', '-')}",
            data="BBBC038v1 (2018 Data Science Bowl), stage1_train",
            data_source="https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
            data_licence="CC0 1.0",
            weights_licence="MIT",
            comparison=row,
        )
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
