"""Is a pretrained encoder worth it, on your data?

    python examples/compare_pretrained_encoders.py --train train.csv --validation val.csv

The answer in this package's own measurement was "not for the score, yes for the epochs" - see
the table in the README. But that was three public datasets, and the regime where transfer
learning helps is narrow enough that the only reliable answer is the one measured on the images
in front of you. This script is that measurement, parameterised.

Four configurations, because the eight that were tried first collapsed to four useful ones:

    built-in                  our encoder, the baseline anything else has to beat
    <backbone> from scratch   the same capacity, no transferred weights - which separates
                              "a bigger encoder helps" from "ImageNet helps"
    pretrained + freeze       hold the transferred layers while the decoder settles
    pretrained + enc lr/10    give them a tenth of the rate instead

The pair in the middle is the point. Without it a win for the pretrained run cannot be told
apart from a win for nine million parameters, and those have different consequences: one means
install an extra, the other means a longer epoch forever.

Several seeds per configuration, not one. The differences being looked for here are smaller than
the spread between seeds on every dataset tried, and a single run of each would rank noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pyplatypus import Engine, from_dict, plot_masks, summarise_cases


def configurations(backbone: str, rate: float, freeze: int) -> list[tuple[str, dict]]:
    pretrained = {"encoder": backbone, "pretrained": True}
    return [
        ("built-in (baseline)", {}),
        (f"{backbone} from scratch", {"encoder": backbone}),
        (f"pretrained, freeze {freeze}", {**pretrained, "freeze_encoder": freeze}),
        ("pretrained, encoder lr/10", {**pretrained, "encoder_learning_rate": rate / 10}),
    ]


def specification(arguments, seed: int, extra: dict) -> dict:
    """The experiment as the configuration the engine takes.

    A function rather than a literal inside `run` so a test can validate it without
    training anything - these scripts went three days carrying a field the spec had
    stopped accepting, because nothing ever read them.
    """
    return {
        "seed": seed,
        "task": "semantic_segmentation",
        "data": {
            "train_path": arguments.train,
            "validation_path": arguments.validation,
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "mode": arguments.mode,
        },
        "models": [
            {
                "name": "m",
                "architecture": "u_net",
                "input_shape": [arguments.size, arguments.size],
                "channels": arguments.channels,
                "blocks": 4,
                "filters": 16,
                "batch_size": arguments.batch,
                "epochs": arguments.epochs,
                "loss": {"name": arguments.loss},
                "metrics": [{"name": "dice", "include_background": False}],
                "optimizer": {"name": "adam", "learning_rate": arguments.rate},
                # Early stopping on the metric, with the best weights restored, so each
                # configuration is judged at its best rather than at whatever epoch the loop
                # happened to stop on. Without it the comparison measures how fast each one
                # overfits, which is a different question.
                "callbacks": [
                    {
                        "name": "early_stopping",
                        "monitor": "val_dice",
                        "patience": arguments.patience,
                        "restore_best": True,
                    }
                ],
                **extra,
            }
        ],
    }


def run(arguments, seed: int, extra: dict, figure: Path | None = None) -> tuple:
    spec = from_dict(specification(arguments, seed, extra))
    engine = Engine(spec, num_workers=arguments.workers)
    history = engine.fit(verbose=False)["m"]
    distribution = {row["metric"]: row for row in summarise_cases(engine.evaluate_cases("m"))}
    best = max(record["val_dice"] for record in history.records)

    if figure is not None:
        # One arm, one figure, at one seed - not eight times three. The table is what ranks
        # these configurations and a picture cannot; what a picture shows is the thing the
        # table only implies, which is that a frozen ImageNet encoder on microscopy produces
        # masks of the wrong shape rather than merely a lower number.
        which = [0, 1, 2, 3]
        dataset = engine.dataset(engine.runs["m"].spec, "validation")
        drawing = plot_masks(
            np.stack([dataset[index][0] for index in which]),
            prediction=engine.predict("m", "validation")[which],
            truth=np.stack([dataset[index][1] for index in which]),
        )
        drawing.savefig(figure, dpi=110, bbox_inches="tight")
        print(f"      wrote {figure}")

    return distribution["dice"], best, len(history.records), engine.runs["m"].parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", required=True, help="training CSV or directory")
    parser.add_argument("--validation", required=True, help="validation CSV or directory")
    parser.add_argument("--mode", default="config_file", choices=["config_file", "nested_dirs"])
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150, help="a cap; early stopping decides")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--freeze", type=int, default=5)
    parser.add_argument("--loss", default="cce_dice")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="measurements")
    parser.add_argument("--figures", default=None, help="a directory to draw each arm's masks into")
    arguments = parser.parse_args()

    rows = []
    for label, extra in configurations(arguments.backbone, arguments.rate, arguments.freeze):
        means, worst, bests, epochs, parameters = [], [], [], [], None
        started = time.time()
        for seed in arguments.seeds:
            drawing = None
            if arguments.figures and seed == arguments.seeds[0]:
                drawing = Path(arguments.figures) / f"encoders-{label.replace(' ', '-')}.png"
                drawing.parent.mkdir(parents=True, exist_ok=True)
            dice, best, ran, parameters = run(arguments, seed, extra, drawing)
            means.append(dice["mean"])
            worst.append(dice["min"])
            bests.append(best)
            epochs.append(ran)
            print(
                f"    {label:28} seed {seed}: dice {dice['mean']:.4f}  "
                f"worst {dice['min']:.3f}  best val {best:.4f}  {ran} epochs",
                flush=True,
            )
        rows.append(
            {
                "config": label,
                "mean": statistics.mean(means),
                # With one seed this is undefined, and that is worth saying rather than hiding:
                # the whole reason for several seeds is that the spread is the thing to compare
                # a difference against.
                "seed_sd": statistics.stdev(means) if len(means) > 1 else None,
                "runs": means,
                "worst": statistics.mean(worst),
                "best_val": statistics.mean(bests),
                "epochs": epochs,
                "parameters": parameters,
                "seconds": time.time() - started,
            }
        )
        spread = rows[-1]["seed_sd"]
        print(
            f"{label:30} dice {rows[-1]['mean']:.4f}"
            f"{f' +/- {spread:.4f}' if spread is not None else ' (one seed)'}"
            f"  epochs {epochs}  {rows[-1]['seconds']:.0f}s",
            flush=True,
        )

    print("\n" + "=" * 96)
    for row in sorted(rows, key=lambda r: -r["mean"]):
        spread = f"+/- {row['seed_sd']:.4f}" if row["seed_sd"] is not None else "(one seed)"
        print(
            f"{row['config']:30} {row['mean']:.4f} {spread:12} "
            f"{row['parameters']:>10,}  epochs {row['epochs']}"
        )
    print("=" * 96)

    best = max(rows, key=lambda r: r["mean"])
    baseline = rows[0]
    gap = best["mean"] - baseline["mean"]
    spreads = [r["seed_sd"] for r in rows if r["seed_sd"] is not None]
    print(f"\nbest: {best['config']} ({best['mean']:.4f})")

    # One seed gives no spread, and treating a missing spread as zero would make every
    # difference look significant - a confident ranking from no evidence about variance,
    # which is worse than declining to rank. So decline.
    if not spreads:
        print(f"over the baseline by {gap:+.4f}, against an unknown seed spread.")
        print("-> one seed per configuration cannot rank these. On every dataset this was")
        print("   tried on, the spread between seeds of one configuration was larger than")
        print("   the difference between configurations. Re-run with --seeds 1 2 3.")
        out = Path(arguments.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "pretrained-comparison.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {path}")
        return

    noise = max(spreads)
    print(f"over the baseline by {gap:+.4f}, against a worst seed spread of {noise:.4f}")
    if abs(gap) <= noise:
        print("-> inside the noise. On this data the encoder is not where the result comes from,")
        print("   and the cheaper model is the answer. Compare the epoch counts instead: reaching")
        print("   the same score sooner is a real saving even when the score does not move.")
    else:
        print("-> outside the seed spread, so it is worth the extra parameters on this data.")
        print("   Check the 'from scratch' row before crediting ImageNet: if it is close to the")
        print("   pretrained one, what helped was capacity, not the transferred weights.")

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pretrained-comparison.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
