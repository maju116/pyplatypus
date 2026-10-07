"""Train YOLOv3 on BCCD through a specification, and say what it scores.

    python examples/detect_blood_cells.py --data path/to/BCCD --epochs 150

BCCD is 364 blood-smear photographs with 4888 boxes in three classes - red cells, white
cells and platelets - from `github.com/Shenggan/BCCD_Dataset`, MIT licence, with its own
train/val/test split. **Taken from that repository rather than the Kaggle mirror**: the
images are the same, but the Kaggle copy is governed by competition rules accepted at
download, and a trained model is a derivative of the data behind it.

This script exists because it is the first point at which anything about detection in this
package is demonstrable rather than merely tested. The metrics, the encoder and the anchors
were each verified against something external; none of that says a model trained with them
finds a blood cell.

**It was written twice.** The first version assembled a detector by hand out of
`pyplatypus.detection` - a Dataset class, a training loop, a decode-and-score function, and
twenty command-line flags for the settings. Everything it did now lives in the engine and
the specification, and what is left here is the part that is genuinely about BCCD: where
its files are, and what its canonical splits say. That is the test of whether the
specification earned its place.

Three things about this dataset decide how it has to be run, and all three were measured
rather than assumed:

- **The classes are wildly unbalanced**: 4155 red cells against 372 white and 361
  platelets. Average precision is reported per class, so a mean over the three is not
  dominated by the red cells - but a single number would be, which is why `evaluate()`
  carries no overall precision or recall and `evaluate_classes()` exists.
- **Two of the 4888 annotations are points rather than boxes** - `xmin == xmax` - which
  Pascal VOC's 1-based reading turns into 1x1 pixels. They are not cells; they are a click
  without a drag. The dataset drops them once the letterbox has shrunk them below a pixel,
  and the survey counts them rather than hiding them.
- **The encoder can lose objects, and on this dataset it barely does.** Two cells of one
  shape whose centres fall in one grid cell share a slot, so the second is dropped. A
  synthetic test that packed 45 equal boxes into a 416 frame lost 9% of them, and that
  number does not transfer: BCCD averages about fourteen objects an image, and the real
  loss at 416 is **3 boxes out of 2804**. `fit()` surveys the targets before the first
  epoch, so a dataset where it does matter can be noticed in time.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

import pyplatypus as pp

LABELS = ["RBC", "WBC", "Platelets"]
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def split_names(root: Path, which: str) -> list[str]:
    """BCCD's own split, so the numbers are comparable to anybody else's on it."""
    listing = root / "ImageSets" / "Main" / f"{which}.txt"
    if not listing.exists():
        raise SystemExit(
            f"{listing} is missing. Point --data at the BCCD directory of "
            f"github.com/Shenggan/BCCD_Dataset, which carries the canonical splits."
        )
    return [line.strip() for line in listing.read_text().splitlines() if line.strip()]


def write_splits(root: Path, out_dir: Path) -> dict[str, Path]:
    """BCCD's layout into three CSV files a specification can point at.

    This is the only part of the script that knows anything about how BCCD is arranged,
    and it is here rather than in the package because every detection dataset is arranged
    its own way. BCCD keeps images in `JPEGImages/`, annotations in `Annotations/`, and
    its splits as lists of names in `ImageSets/Main/` - the Pascal VOC convention.

    `split_dataset()` in the package would also produce three CSVs, but it would *make up*
    the split. BCCD's own is the whole point: a number measured on a split nobody else
    uses cannot be compared with anybody, which is what this script is for.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, listed in SPLIT_FILES.items():
        path = out_dir / f"{split}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            # `annotations`, not `masks`: a detection specification looks for that column,
            # because what labels an image is an annotation file and not a picture.
            writer.writerow(["key", "images", "annotations"])
            for name in split_names(root, listed):
                writer.writerow(
                    [
                        name,
                        str((root / "JPEGImages" / f"{name}.jpg").resolve()),
                        str((root / "Annotations" / f"{name}.xml").resolve()),
                    ]
                )
        written[split] = path
    return written


def configuration(splits: dict[str, Path], arguments) -> dict:
    """The experiment, as the dict a YAML file would have parsed into.

    Written as data rather than built with keyword arguments so that the same thing can
    be dumped to a file and handed to someone: the point of the specification is that a
    configuration file and a script describe the same run, and this is where that is
    cheapest to demonstrate.
    """
    return {
        "task": "object_detection",
        "seed": arguments.seed,
        "data": {
            "mode": "config_file",
            "train_path": str(splits["train"]),
            "validation_path": str(splits["validation"]),
            "test_path": str(splits["test"]),
            "classes": LABELS,
            "annotation_format": "pascal_voc",
        },
        "models": [
            {
                "name": "bccd",
                "architecture": "yolo3",
                "input_shape": [arguments.size, arguments.size],
                "anchors_per_grid": arguments.anchors_per_grid,
                "box_loss": arguments.box_loss,
                "epochs": arguments.epochs,
                "batch_size": arguments.batch,
                "score_threshold": arguments.objectness,
                "operating_point": arguments.operating_point,
                "nms_threshold": arguments.nms_iou,
                "optimizer": {"name": "adam", "learning_rate": arguments.rate},
                # Horizontal flip only. A smear has no up or down, and a left-right flip is
                # free: the boxes follow it exactly, with no interpolation and no rounding.
                # 205 training images is few enough that seeing each one both ways round
                # matters - measured, see the note at the top of `report`.
                "augmentation": [{"name": "HorizontalFlip", "params": {"p": 0.5}}],
                # And the rate decays to nothing over the run. Worth about +0.006 of
                # mAP@[.50:.95] here, from one run with and one without - which is the
                # honest size of it and not a clean measurement; see DETECTION_RECON.md §12.
                "callbacks": [{"name": "cosine_annealing"}],
            }
        ],
    }


def report(engine, split: str) -> dict:
    """The table, printed the way a reader wants it and returned the way a file wants it.

    One `engine.report(...)` rather than `evaluate()` and `evaluate_classes()`, which are
    both views over it: calling the two would run the model over the split twice for the
    same numbers.
    """
    measured = engine.report("bccd", split)
    model = measured.as_row(engine.runs["bccd"])
    per_class = measured.per_class()

    print(f"=== {split} ===")
    print(f"  mAP@0.5       {model['map_50']:.4f}")
    print(f"  mAP@[.50:.95] {model['map_50_95']:.4f}")
    # The gap between those two is localisation, and this is it as one number: how well
    # the boxes that matched actually fit, rather than merely that they cleared 0.5.
    if model["mean_matched_iou"] is not None:
        print(f"  mean IoU of matched boxes {model['mean_matched_iou']:.4f}")
    print(
        f"  {'class':12} {'AP@0.5':>8} {'IoU':>6} {'truth':>6} {'pred':>7} {'prec':>7} {'rec':>7}"
    )
    for row in per_class:
        print(
            f"  {row['class']:12} {_f(row['average_precision'], 4):>8} "
            f"{_f(row['mean_matched_iou'], 3):>6} {row['n_truth']:>6} "
            f"{row['n_predicted']:>7} {_f(row['precision'], 3):>7} "
            f"{_f(row['recall'], 3):>7}"
        )
    print()

    # Which frames, not just how well on average. A mean over a split says 0.86; it does
    # not say that the misses are concentrated in a handful of images - and if they are,
    # that is usually something about the data rather than the model. This runs the model
    # over the split a second time, which on BCCD's 72 test images is seconds; on a large
    # split it is a forward pass to budget for.
    images = engine.evaluate_images("bccd", split)
    worst = sorted(images, key=lambda row: (-row["missed"], row["mean_matched_iou"] or 0))
    print(f"  worst {min(5, len(worst))} images by boxes missed")
    print(f"  {'image':28} {'truth':>6} {'found':>6} {'missed':>7} {'extra':>6} {'IoU':>6}")
    for row in worst[:5]:
        print(
            f"  {str(row['key'])[:28]:28} {row['n_truth']:>6} {row['matched']:>6} "
            f"{row['missed']:>7} {row['spurious']:>6} "
            f"{_f(row['mean_matched_iou'], 3):>6}"
        )
    clean = sum(1 for row in images if not row["missed"] and not row["spurious"])
    print(f"  {clean} of {len(images)} images exactly right\n")

    return {"model": model, "per_class": per_class, "per_image": images}


def _f(value, places: int) -> str:
    return f"{value:.{places}f}" if value is not None else "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="the BCCD directory")
    parser.add_argument("--size", type=int, default=416, help="divisible by 32")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--batch", type=int, default=8, help="the micro-batch that has to fit in memory"
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="micro-batches per optimiser step, so the effective batch is "
        "--batch times this. A 608 input at batch 8 needs 7.4 GB of "
        "activations, which does not fit an 8 GB card; 4 with 2 "
        "accumulated does, at the same effective batch. **Not the "
        "same run**: batch norm still sees the micro-batch, so its "
        "statistics are noisier than a true batch of 8.",
    )
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--anchors-per-grid", type=int, default=3)
    parser.add_argument(
        "--box-loss",
        choices=("offsets", "giou"),
        default="offsets",
        help="how the box coordinates are scored. 'offsets' is YOLOv3's "
        "own and is what the published weights were trained with; "
        "'giou' scores the decoded box directly and can reach zero",
    )
    parser.add_argument(
        "--objectness",
        type=float,
        default=0.01,
        help="kept low: average precision integrates the whole ranking",
    )
    parser.add_argument(
        "--operating-point",
        type=float,
        default=0.5,
        help="the confidence at which precision and recall are reported",
    )
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--figures", default=None, help="a directory to draw the boxes and the anchors into"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="measurements")
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH.safetensors",
        help="where to write the trained weights, as safetensors with a "
        "sidecar carrying the anchors - which the weights are "
        "meaningless without. Any other suffix is replaced. Without "
        "this the run cannot be asked anything afterwards",
    )
    arguments = parser.parse_args()

    root = Path(arguments.data)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = write_splits(root, out / "bccd_splits")
    config = configuration(splits, arguments)

    # The same experiment as a file, beside the results. Not decoration: it is the claim
    # that the two interfaces are one pipeline, in a form a reader can check by running it.
    (out / "detection-bccd.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    spec = pp.from_dict(config)
    engine = pp.build_engine(spec, num_workers=arguments.workers, accumulate=arguments.accumulate)
    counts = engine.split_sizes()
    print("BCCD: " + ", ".join(f"{n} {name}" for name, n in counts.items()))
    effective = arguments.batch * arguments.accumulate
    print(
        f"input {arguments.size}, micro-batch {arguments.batch}"
        + (
            f" x {arguments.accumulate} accumulated = {effective}"
            if arguments.accumulate > 1
            else ""
        )
    )

    started = time.time()
    history = engine.fit(verbose=True)
    print(f"\ntrained in {(time.time() - started) / 60:.1f} min")

    run = engine.runs["bccd"]
    print(f"model: {run.parameters:,} parameters")
    for row in run.anchor_fit.as_rows():
        print(
            f"  grid {row['grid']} slot {row['slot']}: "
            f"{row['width_pixels']:6.1f} x {row['height_pixels']:6.1f} px, "
            f"{row['boxes']:4} boxes, IoU {row['mean_iou']:.3f}"
        )
    # Asked of validation on purpose. Anchors fitted to the training boxes cover those
    # boxes well by construction; whether they cover a split they were not fitted on is
    # the question, and a large gap says the two splits hold different objects.
    print(f"  anchor coverage on validation: {engine.anchor_coverage('bccd', 'validation')}\n")

    if arguments.save:
        print(f"weights written to {engine.export_weights('bccd', arguments.save)}\n")

    if arguments.figures:
        figures = Path(arguments.figures)
        figures.mkdir(parents=True, exist_ok=True)

        # The anchors against the shapes they were fitted to, asked of the engine rather
        # than assembled here. Widths from one place and anchors from another is how a
        # figure comes to show boxes in different places from where the anchors were fitted
        # to them, which looks like a bad fit and is a bug. Drawn for the split they were
        # *not* fitted on as well: that is the only way to see a class they have nothing
        # near.
        for split in ("train", "validation"):
            path = figures / f"bccd-anchors-{split}.png"
            pp.plot_anchors(engine, split=split).savefig(path, dpi=110, bbox_inches="tight")
            print(f"  wrote {path}")

        # One frame, not a montage: `plot_boxes` stacks vertically by design, so three
        # images make a column nobody reads. And the frame is chosen by reading the
        # annotations rather than by eye - a picture of red cells alone shows a third of
        # what the detector does.
        found = engine.predict("bccd", "test")
        dataset = engine.dataset(engine.runs["bccd"].spec, "test")
        annotations = dataset.annotations()
        wanted = [
            index
            for index, annotation in enumerate(annotations)
            if set(engine.spec.data.classes) <= set(annotation.names)
        ]
        if not wanted:
            print("  no test frame holds all three classes; drawing the first instead")
        show = wanted[0] if wanted else 0

        # At the file's own size: the boxes come back in the source image's pixels, and a
        # box drawn over a resized copy sits beside the cell it belongs to.
        images = np.stack([pp.read_image(dataset.samples[show].images[0], size=None)])
        figure = pp.plot_boxes(
            images,
            [found[show]],
            truth=[annotations[show].as_truth()],
            labels=[Path(dataset.samples[show].images[0]).name],
        )
        path = figures / "bccd-boxes.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  wrote {path}")

        # Printed so a caption is transcribed rather than counted off the picture.
        drawn = [
            name
            for name, score in zip(found[show]["names"], found[show]["scores"], strict=True)
            if score >= 0.5
        ]
        counted = {name: drawn.count(name) for name in engine.spec.data.classes}
        truth_counted = {
            name: list(annotations[show].names).count(name) for name in engine.spec.data.classes
        }
        print(f"  drawn at min_score 0.5: {counted}, annotated: {truth_counted}")

    results = {split: report(engine, split) for split in ("validation", "test")}
    (out / "bccd-yolo3.json").write_text(
        json.dumps(
            {
                "configuration": config,
                "anchors": [[list(pair) for pair in group] for group in run.anchors],
                "anchor_mean_iou": run.anchor_fit.mean_iou if run.anchor_fit else None,
                "targets": run.survey.to_dict() if run.survey else None,
                "history": history["bccd"].records,
                "validation": results["validation"],
                "test": results["test"],
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(f"wrote {out / 'bccd-yolo3.json'}")


if __name__ == "__main__":
    main()
