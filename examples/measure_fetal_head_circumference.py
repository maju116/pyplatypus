"""A head circumference in millimetres: the one example that does not end in a score.

    python examples/measure_fetal_head_circumference.py --data examples/data/hc18

Every other example here ends with a Dice, an IoU or an average precision. **A sonographer
writes down none of those.** They write down a head circumference in millimetres, and
whether the mask was good is a question they never ask out loud. This example ends with the
number a person acts on, and measures the error in the unit that number is written in.

## What is measured, and in which order

Three things, and the first two need no model at all - which is the point of doing them
first:

1. **The measurement pipeline against the sonographer's own ellipse.** Fit the ellipse to
   the *truth* mask, take its circumference, convert with the image's own pixel size, and
   compare against the millimetres the dataset ships. Nothing here is about a network: it
   says what the arithmetic costs before any model is trained. A mask is scored against a
   mask; a circumference is scored against a circumference, and the second has an error
   floor the first does not.
2. **What a change invisible to Dice does to the millimetres.** Dilate the truth mask by a
   ring of pixels and the overlap barely moves, because a ring is a small share of a large
   ellipse - while every millimetre of that ring lands in the circumference twice over.
   Then rough the boundary up without moving it on average: Dice falls and the millimetres
   do not. **The two quantities are not two readings of one thing**, and this is where that
   stops being an argument and becomes a table.
3. **A trained model, in both units**, with the per-case ranking by Dice set against the
   per-case ranking by millimetre error.

## Why the ellipse is fitted from the region's moments

A uniform filled ellipse has variance `a^2 / 4` along its major axis, so the second moments
of the mask *are* the semi-axes - exactly, for a true ellipse, with no iteration and no
boundary to trace. Two consequences, and both are the reason this is the right instrument
here rather than a convenience:

* a ragged boundary moves an integral over the region hardly at all, so the fit is doing
  what a sonographer does by eye rather than chasing speckle;
* a mask that is systematically fat or thin moves it linearly, which is the error that
  matters and the one Dice is least able to see.

The circumference of an ellipse has no closed form. Ramanujan's second approximation is
used, and a test holds it to a numerically integrated perimeter - it is right to about one
part in ten million at the eccentricities a fetal skull has, which is four orders of
magnitude below the pixel.

## The data, and the two things the dataset does not say outright

HC18, **CC BY 4.0** - read from the dataset's own Zenodo record (`10.5281/zenodo.1327317`,
176.5 MB, open access) rather than from a mirror. The challenge page names no licence at
all and asks for the paper and the dataset to be cited, which is the attribution the
licence already requires. 999 training images, 335 test, 800 x 540 pixels with a pixel size
ranging from 0.052 to 0.326 mm, annotated by a trained sonographer.

**The test set ships no circumferences.** `test_set_pixel_size.csv` has the pixel size and
nothing else: the reference measurement exists only for the 999 training images, because
the challenge keeps the rest. So the measurement here is made on a split of the training
set, and the alternative - a number from the challenge's own leaderboard - is not a number
this example could have computed or checked.

**The annotation is the ellipse, not the head.** Each `*_Annotation.png` is the one-pixel
curve the sonographer drew: 0.4% of the image is foreground, where a filled skull is
nearer 20%. Training on it directly would be training a model to find a line, so the
outlines are filled once, in `prepare`, and written beside the data. That step is not a
workaround - it is what anyone holding this dataset must do first, and a page that hid it
would be hiding the reason the masks on disk are not the masks in the download.

**Some images are the same examination.** The challenge page says so in as many words -
*"Some ultrasound images were made during the same echoscopic examination and have
therefore a very similar appearance"* - and the dataset ships the grouping without ever
calling it that: 999 images carry only 806 distinct leading numbers, so `000_HC.png` and
`000_2HC.png` are one examination photographed twice. Counted here: 642 examinations with
one image, 141 with two, 17 with three, 6 with four. A split by row therefore puts
near-duplicates on both sides of it for 357 of the 999 images, which is `pyplatypus#200`'s
leak in a second dataset - measured there as several points of Dice given away for free.
`split.group_by` is given `^(\\d+)_` for exactly that reason, and the three-way split means
the millimetres are reported on data that early stopping never saw either.

## No weights are published, and that is a decision rather than an omission

The licence permits it: CC BY 4.0, the same bracket as `fives-unet`. The reason not to is
that the artefact here is a number in millimetres, and a published `hc18-unet` would be
weights measured on a split of the training set that nobody else cut - not comparable with
the leaderboard everyone quotes for this dataset, and comparable with nothing else. If
published HC18 weights are wanted they are their own piece of work, with the challenge's
own test set and a submission.

## Where the numbers are, and why they are not here

Every measured result - the pipeline's own error, the table of what a change invisible to
Dice costs, the trained arm - is printed by this script, written to its `--out` directory as
JSON, and recorded on `maju116/pyplatypus#205`. Deliberately not in this docstring: §3C of
the instructions wants a number in prose held to a measurement by a test, and nothing can
hold a docstring to a run that needs the dataset on disk. What is stated above instead is the
method, the decisions, and facts about the dataset that cannot change - its licence, its
size, how many images and examinations it has - each read from the dataset's own pages on
2026-10-25 rather than from a mirror.

`maju116/pyplatypus#205`. The article that narrates this is `maju116/rplatypus#192`, which
is where the page and its R half live.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

import pyplatypus as pp
from pyplatypus import ellipse_circumference

#: The examination, out of the file name. Everything sharing one must land in one split, and
#: the first capture group is what `group_of` reads.
GROUP_BY = r"^(\d+)_"

#: What `prepare` writes beside the data, and what the run reads.
CASES = "cases.csv"


def reference(path: Path) -> dict[str, dict[str, float]]:
    """`key -> {pixel_mm, hc_mm}`, read from the dataset's own CSV.

    The key is the image's stem, which is what a `config_file` sample is called and
    therefore what `predict_stream` yields back.
    """
    table: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["filename"]
            table[Path(name).stem] = {
                "pixel_mm": float(row["pixel size(mm)"]),
                "hc_mm": float(row["head circumference (mm)"]),
            }
    if not table:
        raise SystemExit(f"{path} holds no rows; see the module docstring")
    return table


def fill(outline: np.ndarray) -> np.ndarray:
    """The curve the sonographer drew, as the region it encloses.

    `binary_fill_holes` fills what the curve surrounds, so a closed outline becomes a disc
    and an *open* one fills nothing at all - which is the failure worth knowing about,
    because it is silent: an unclosed annotation would arrive as an empty mask rather than
    as an error. `prepare` therefore refuses a mask whose area is close to the outline's
    own, which is what an unfilled one looks like.
    """
    from scipy.ndimage import binary_fill_holes

    return np.asarray(binary_fill_holes(outline > 0), dtype=bool)


def circumference_mm(mask: np.ndarray, pixel_mm: float) -> float:
    """The head circumference a mask implies, in millimetres.

    The arithmetic is the engine's as of `0.8.0a39` - `largest_filled_object` for the
    repair, `ellipse_axes` for the second moments, `ellipse_circumference` for Ramanujan's
    perimeter - and this is the one line of it that belongs to the dataset: HC18 records one
    `pixel size(mm)` per image, square, so the spacing is that number twice.

    The pixel size is the image's own, out of the dataset's CSV, which is why this is
    measured on the **source** grid: a prediction resized to the model's shape has pixels of
    a size nobody recorded, and multiplying by the file's number there would be wrong by the
    ratio of the two shapes.
    """
    return ellipse_circumference(mask, (pixel_mm, pixel_mm))


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Foreground Dice of two boolean masks, for the column beside the millimetres."""
    overlap = float(np.logical_and(prediction, truth).sum())
    total = float(prediction.sum() + truth.sum())
    return 1.0 if total == 0.0 else 2.0 * overlap / total


def prepare(root: Path, out: Path) -> tuple[Path, dict[str, dict[str, float]], list[str]]:
    """Fill every outline, write the masks and one CSV naming the usable cases.

    The CSV carries a `key` because a `config_file` sample without one is called `row 55`,
    which cannot be grouped by - and the grouping is half of what this example has to say.

    **Four of the 999 annotations do not close**, and they are left out rather than filled.
    Measured: `186_HC`, `346_HC`, `628_2HC` and `787_HC` have a curve that runs off the edge
    of the image, so the region it would enclose is connected to the outside and
    `binary_fill_holes` returns the curve unchanged - morphological closing up to 9x9 does
    not join them either, because the gap is the width of the frame rather than a pixel.
    They are a real limitation of this instrument and not a defect in the data: the
    sonographer's ellipse continues outside the picture, so the reference millimetres
    describe a shape the file does not contain, and a fit over the second moments of a
    truncated ellipse measures something else. Fitting a conic to the visible arc is how the
    literature handles it, and it is the reason that method exists.
    """
    images = root / "training_set" / "training_set"
    table = reference(root / "training_set_pixel_size_and_HC.csv")
    masks = out / "masks"
    masks.mkdir(parents=True, exist_ok=True)

    rows = ["key,images,masks"]
    open_curves = []
    # Read once rather than per key: this is the kind of file access that turns a cheap
    # loop into a thousand of them.
    already_open = _unusable(out)
    for key in sorted(table):
        image = images / f"{key}.png"
        annotation = images / f"{key}_Annotation.png"
        if not image.exists() or not annotation.exists():
            raise SystemExit(f"{key}: no image or no annotation under {images}")
        target = masks / f"{key}.png"
        if not target.exists():
            outline = np.asarray(pp.read_image(annotation, channels=1, size=None, nearest=True))
            outline = outline.reshape(outline.shape[:2])
            filled = fill(outline)
            if filled.sum() < 2.0 * float((outline > 0).sum()):
                open_curves.append(key)
                continue
            # PIL rather than one of the package's writers: those take a prediction and
            # write a colour map, and this is a binary mask being derived from the data
            # before any model exists.
            Image.fromarray((filled * 255).astype(np.uint8)).save(target)
        elif key in already_open:
            open_curves.append(key)
            continue
        rows.append(f"{key},{image.resolve()},{target.resolve()}")

    if open_curves:
        (out / "open-curves.txt").write_text("\n".join(sorted(open_curves)) + "\n")
    cases = out / CASES
    cases.write_text("\n".join(rows) + "\n")
    usable = len(rows) - 1
    groups = {re.match(GROUP_BY, row.split(",")[0]).group(1) for row in rows[1:]}
    print(
        f"  {usable} of {len(table)} images, {len(groups)} examinations -> {cases}\n"
        f"  {len(open_curves)} annotations left out, their curve open at the frame: "
        f"{', '.join(sorted(open_curves)) or 'none'}"
    )
    return cases, table, sorted(open_curves)


def _unusable(out: Path) -> set[str]:
    """The keys an earlier run found unfillable, so a second run leaves them out too.

    Without this the exclusion would depend on whether the masks directory already
    existed - the one way a cached step can change a measurement.
    """
    path = out / "open-curves.txt"
    return set(path.read_text().split()) if path.exists() else set()


def specification(cases: Path, arguments) -> dict:
    """The run, as the dict a YAML file would have parsed into.

    One path and a three-way `split`, rather than three paths: the division has to be by
    examination and `split.group_by` is where that belongs. The third fraction is what the
    millimetres are reported on, so no number here comes off the data `early_stopping`
    watched.
    """
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(cases),
            "mode": "config_file",
            "colormap": [[0, 0, 0], [255, 255, 255]],
            "split": {
                "fractions": [0.6, 0.2, 0.2],
                # 999 images, 806 examinations. Without this, 357 of them have a
                # near-duplicate on the other side of the division.
                "group_by": GROUP_BY,
                "seed": arguments.seed,
            },
        },
        "models": [
            {
                "name": "skull",
                "architecture": "u_net",
                # 800 x 540 arrives as 800 x 544, which divides by 2^4 and resamples the
                # data almost not at all. A measurement that ends in millimetres should
                # argue with the ellipse, not with the interpolation.
                "input_shape": [arguments.height, arguments.width],
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
                    # No vertical flip: an ultrasound fan has a top and a bottom, and a
                    # mirrored one is a picture of nothing. The ellipse is symmetric under
                    # the horizontal one.
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "RandomBrightnessContrast", "params": {"p": 0.3}},
                ],
            }
        ],
        "seed": arguments.seed,
    }


def _read_mask(path: str) -> np.ndarray:
    array = np.asarray(pp.read_image(path, channels=1, size=None, nearest=True))
    return array.reshape(array.shape[:2]) > 127


def pipeline_control(
    cases: Path, table: dict[str, dict[str, float]], limit: int, images: Path
) -> dict:
    """What the measurement costs before any model exists, and where that cost comes from.

    The truth mask through the same ellipse fit, the same circumference and the same pixel
    size, against the millimetres the sonographer recorded. This is the floor: no model can
    be read as better than the arithmetic that measures it, and a figure quoted without it
    is a figure whose error could be entirely our own.

    It also explains itself, which is the part worth more than the number. The floor came
    out **biased rather than noisy** - every case too large by about the same amount - and
    the cause is the annotation's own line. The curve a sonographer drew is 2.5 pixels wide,
    not one; filling it puts the region's boundary half a width outside the curve, and a
    radius larger by `w / 2` makes a circumference larger by `pi * w`. So the prediction is
    `pi * w * pixel_mm` per case, with `w` measured as the outline's area over the fitted
    ellipse's perimeter, and it is reported beside the bias rather than subtracted from it:
    a correction built into the measurement would hide the one number here that is about the
    dataset rather than about us.
    """
    rows = []
    with cases.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= limit:
                break
            key = row["key"]
            # In pixels, because the line width this measures is a pixel count: the
            # annotation's own stroke, divided out of it below.
            length = ellipse_circumference(_read_mask(row["masks"]))
            outline = np.asarray(
                pp.read_image(images / f"{key}_Annotation.png", channels=1, size=None, nearest=True)
            )
            width = float((outline.reshape(outline.shape[:2]) > 0).sum()) / length
            rows.append(
                {
                    "case": key,
                    "mm": length * table[key]["pixel_mm"],
                    "reference": table[key]["hc_mm"],
                    "width_px": width,
                    "predicted_mm": math.pi * width * table[key]["pixel_mm"],
                }
            )
    errors = np.array([row["mm"] - row["reference"] for row in rows])
    predicted = np.array([row["predicted_mm"] for row in rows])
    widths = np.array([row["width_px"] for row in rows])
    residual = errors - predicted
    return {
        "n": len(rows),
        "mae_mm": float(np.abs(errors).mean()),
        "bias_mm": float(errors.mean()),
        "sd_mm": float(errors.std(ddof=1)),
        "worst_mm": float(np.abs(errors).max()),
        "relative": float(np.abs(errors / np.array([row["reference"] for row in rows])).mean()),
        "line_width_px": float(widths.mean()),
        "line_width_sd_px": float(widths.std(ddof=1)),
        "predicted_bias_mm": float(predicted.mean()),
        "correlation": float(np.corrcoef(predicted, errors)[0, 1]),
        "residual_mae_mm": float(np.abs(residual).mean()),
        "residual_bias_mm": float(residual.mean()),
    }


def perturbation_control(
    cases: Path, table: dict[str, dict[str, float]], limit: int, radii: list[int], seed: int
) -> list[dict]:
    """Dice and millimetres under changes to the truth, so the two can be compared directly.

    Three kinds, and the pairing is the finding. A dilation and an erosion move the boundary
    outwards and inwards by a known number of pixels; the ragged arm toggles pixels in a band
    across the boundary, so the boundary moves everywhere and nowhere on average. One of
    these is what a slightly over-confident model does and one is what a noisy one does, and
    Dice reports them the wrong way round.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    rng = np.random.default_rng(seed)
    truths = []
    with cases.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= limit:
                break
            truths.append((row["key"], _read_mask(row["masks"])))

    out = []
    for kind in ("dilated", "eroded", "ragged"):
        for radius in radii:
            dices, shifts = [], []
            for key, truth in truths:
                if kind == "dilated":
                    changed = binary_dilation(truth, iterations=radius)
                elif kind == "eroded":
                    changed = binary_erosion(truth, iterations=radius)
                else:
                    band = binary_dilation(truth, iterations=radius) ^ binary_erosion(
                        truth, iterations=radius
                    )
                    changed = truth ^ (band & (rng.random(truth.shape) < 0.5))
                pixel = table[key]["pixel_mm"]
                dices.append(dice(changed, truth))
                shifts.append(circumference_mm(changed, pixel) - circumference_mm(truth, pixel))
            out.append(
                {
                    "kind": kind,
                    "radius": radius,
                    "dice": float(np.mean(dices)),
                    "mm": float(np.mean(shifts)),
                    "abs_mm": float(np.mean(np.abs(shifts))),
                }
            )
    return out


def measure(engine, cases: Path, table: dict[str, dict[str, float]], split: str) -> list[dict]:
    """One row per case: Dice and millimetres, both on the grid the files are on.

    `space="source"` is what makes the second column meaningful. The pixel size is a
    property of the file, so a prediction measured on the model's grid would be multiplied
    by a number belonging to a different grid - and streamed rather than stacked because
    the answer is at full resolution.
    """
    paths = {}
    with cases.open(newline="") as handle:
        for row in csv.DictReader(handle):
            paths[row["key"]] = row["masks"]

    rows = []
    for case, prediction in engine.predict_stream("skull", split=split, space="source"):
        mask = np.argmax(prediction, axis=-1) > 0
        truth = _read_mask(paths[case])
        pixel = table[case]["pixel_mm"]
        got = circumference_mm(mask, pixel)
        rows.append(
            {
                "case": case,
                "dice": dice(mask, truth),
                "mm": got,
                "reference": table[case]["hc_mm"],
                "error_mm": got - table[case]["hc_mm"],
                "truth_mm": circumference_mm(truth, pixel),
                "pixel_mm": pixel,
            }
        )
    return rows


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(values)))
    return order.astype(np.float64)


def disagreement(rows: list[dict]) -> dict:
    """How differently the two columns order the same cases.

    Spearman by hand rather than by importing scipy.stats for one number: the ranks are
    what the coefficient is, and a correlation computed on them is the same thing.
    """
    by_dice = _rank([row["dice"] for row in rows])
    by_error = _rank([-abs(row["error_mm"]) for row in rows])
    correlation = float(np.corrcoef(by_dice, by_error)[0, 1])
    best = sorted(rows, key=lambda row: -row["dice"])[: max(1, len(rows) // 10)]
    return {
        "spearman": correlation,
        "best_decile_by_dice": {
            "n": len(best),
            "dice": float(np.mean([row["dice"] for row in best])),
            "mae_mm": float(np.mean([abs(row["error_mm"]) for row in best])),
            "worst_mm": float(np.max([abs(row["error_mm"]) for row in best])),
        },
    }


def summary(rows: list[dict]) -> dict:
    """The two columns, and the model's error separated from the dataset's own offset.

    `mae_mm` is against the millimetres a sonographer recorded, which is what somebody
    acting on the number would see - and it carries the +0.7 mm the pipeline control
    attributes to the width of the drawn line, because the model was trained on masks made
    by filling that line. `model_mae_mm` is against the circumference of the **truth mask**
    instead, which is the same arithmetic on both sides and therefore the model's own
    contribution. Reporting only the first would blame the model for a drawing convention;
    reporting only the second would hide the convention from the person reading millimetres.
    """
    errors = np.array([row["error_mm"] for row in rows])
    own = np.array([row["mm"] - row["truth_mm"] for row in rows])
    return {
        "n": len(rows),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "dice_worst": float(np.min([row["dice"] for row in rows])),
        "mae_mm": float(np.abs(errors).mean()),
        "bias_mm": float(errors.mean()),
        "sd_mm": float(errors.std(ddof=1)),
        "worst_mm": float(np.abs(errors).max()),
        "model_mae_mm": float(np.abs(own).mean()),
        "model_bias_mm": float(own.mean()),
        "model_worst_mm": float(np.abs(own).max()),
    }


def figure(rows: list[dict], perturbations: list[dict], floor: float, path: Path) -> None:
    """Two panels: the mechanism, then the same thing on real predictions.

    Left, what a known change to the truth does to each quantity - Dice along the bottom,
    millimetres up the side, one curve per kind of change. The dilated and eroded curves sit
    on top of each other, which is itself the point, so only one of them is labelled with
    the radius.

    Right, one point per test case, **on a log scale**. The first version of this panel used
    a linear one and three catastrophic cases at 60 to 96 mm flattened the other 196 into a
    line along the axis - a figure that showed the outliers and hid the finding. The
    quadrant lines are drawn with their counts, because the claim is a count: how many cases
    sit at a Dice anybody would publish and a circumference nobody would accept.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyplatypus.style import CLASS_COLOURS

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for index, kind in enumerate(("dilated", "eroded", "ragged")):
        points = [row for row in perturbations if row["kind"] == kind]
        left.plot(
            [row["dice"] for row in points],
            [row["abs_mm"] for row in points],
            marker="o",
            color=CLASS_COLOURS[index],
            label=kind,
        )
        if kind == "eroded":
            continue  # it lies under `dilated`; one set of radius labels, not two
        for row in points:
            left.annotate(
                f"{row['radius']} px",
                (row["dice"], row["abs_mm"]),
                textcoords="offset points",
                xytext=(6, -3) if kind == "dilated" else (6, 4),
                fontsize=7,
                color=CLASS_COLOURS[index],
            )
    left.axhline(floor, color=CLASS_COLOURS[7], linestyle="--", linewidth=1)
    left.annotate(
        f"the measurement's own error, {floor:.2f} mm",
        (0.02, floor),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=7,
        color=CLASS_COLOURS[7],
    )
    left.set_xlabel("Dice against the truth it was made from")
    left.set_ylabel("absolute circumference error (mm)")
    left.set_title("A known change to the truth, in both units")
    left.legend(frameon=False, fontsize=8, loc="upper right")

    errors = [max(abs(row["error_mm"]), 1e-3) for row in rows]
    dices = [row["dice"] for row in rows]
    right.scatter(dices, errors, s=16, alpha=0.6, color=CLASS_COLOURS[0], edgecolors="none")
    right.set_yscale("log")
    right.axhline(floor, color=CLASS_COLOURS[7], linestyle="--", linewidth=1)

    cut_dice, cut_mm = 0.95, 2.0
    right.axvline(cut_dice, color=CLASS_COLOURS[7], linewidth=0.8, alpha=0.6)
    right.axhline(cut_mm, color=CLASS_COLOURS[7], linewidth=0.8, alpha=0.6)
    publishable = [(d, e) for d, e in zip(dices, errors, strict=True) if d >= cut_dice]
    costly = [pair for pair in publishable if pair[1] > cut_mm]
    right.annotate(
        f"{len(costly)} of {len(publishable)} cases at Dice >= {cut_dice}\nare over {cut_mm:.0f} mm",
        # Bottom left: the only empty corner, because a case with a poor Dice and an
        # excellent circumference is rare - which is itself worth seeing.
        (0.03, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        color=CLASS_COLOURS[7],
    )
    right.set_xlabel("Dice")
    right.set_ylabel("absolute circumference error (mm), log scale")
    right.set_title(f"The trained model, {len(rows)} held-out images")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="examples/data/hc18")
    parser.add_argument("--out", default="measurements/hc18")
    parser.add_argument("--height", type=int, default=544, help="divisible by 2^blocks")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--loss", default="cce_dice")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--controls",
        type=int,
        default=200,
        help="how many truth masks the two no-model controls read",
    )
    parser.add_argument("--figures", default=None, help="a directory to draw the figure into")
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="the two controls that need no model, and then stop",
    )
    parser.add_argument(
        "--redraw",
        default=None,
        help="rebuild the figure from a previous run's JSON and stop, with no training",
    )
    arguments = parser.parse_args()

    if arguments.redraw:
        # A figure is edited more often than a measurement is repeated, and the two should
        # not be the same twenty minutes of GPU. Everything drawn comes off the record.
        record = json.loads(Path(arguments.redraw).read_text())
        figure(
            record["cases"],
            record["perturbation_control"],
            record["pipeline_control"]["mae_mm"],
            Path(arguments.figures or "assets") / "article-hc18-millimetres.png",
        )
        return

    root = Path(arguments.data)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    print("--- filling the outlines ---")
    cases, table, open_curves = prepare(root, out)

    print("\n--- the measurement pipeline against the sonographer's own ellipse ---")
    floor = pipeline_control(
        cases, table, arguments.controls, root / "training_set" / "training_set"
    )
    print(
        f"  n={floor['n']}  mae {floor['mae_mm']:.3f} mm  bias {floor['bias_mm']:+.3f} mm  "
        f"sd {floor['sd_mm']:.3f}  worst {floor['worst_mm']:.3f}  "
        f"({floor['relative'] * 100:.3f}% of the circumference)"
    )
    print(
        f"  the bias is the annotation's line: {floor['line_width_px']:.3f} px wide "
        f"(sd {floor['line_width_sd_px']:.3f}), so pi * w * pixel predicts "
        f"{floor['predicted_bias_mm']:+.3f} mm against {floor['bias_mm']:+.3f} measured, "
        f"correlation {floor['correlation']:.4f}; what is left is "
        f"{floor['residual_mae_mm']:.3f} mm"
    )

    print("\n--- what a change invisible to Dice does to the millimetres ---")
    perturbations = perturbation_control(
        cases, table, arguments.controls, [1, 2, 3, 5], arguments.seed
    )
    print(f"  {'change':10s} {'dice':>8s} {'mm':>9s} {'|mm|':>8s}")
    for row in perturbations:
        print(
            f"  {row['kind'] + ' ' + str(row['radius']) + 'px':10s} {row['dice']:8.4f} "
            f"{row['mm']:+9.3f} {row['abs_mm']:8.3f}"
        )

    record = {
        "open_curves": open_curves,
        "pipeline_control": floor,
        "perturbation_control": perturbations,
    }
    if arguments.controls_only:
        (out / "hc18-controls.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
        print(f"\nwrote {out / 'hc18-controls.json'}")
        return

    config = specification(cases, arguments)
    spec = pp.from_dict(config)
    engine = pp.build_engine(spec, num_workers=arguments.workers)

    started = time.time()
    history = engine.fit(verbose=True)["skull"]
    minutes = (time.time() - started) / 60
    print(f"\ntrained in {minutes:.1f} min, {len(history.records)} epochs")

    print("\n--- the split as one number, on the grid the files are on ---")
    for row in engine.evaluate(split="test", at="source"):
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    print("\n--- per case, in both units ---")
    rows = measure(engine, cases, table, "test")
    totals = summary(rows)
    print(
        f"  n={totals['n']}  dice {totals['dice']:.4f} (worst {totals['dice_worst']:.4f})  "
        f"mae {totals['mae_mm']:.3f} mm  bias {totals['bias_mm']:+.3f} mm  "
        f"worst {totals['worst_mm']:.3f} mm"
    )
    print(
        f"  against the truth mask's own circumference, which is the model's share of it: "
        f"mae {totals['model_mae_mm']:.3f} mm  bias {totals['model_bias_mm']:+.3f} mm  "
        f"worst {totals['model_worst_mm']:.3f} mm"
    )

    # The control for this file's own reading of the prediction: the engine scores the same
    # split on the same grid, by its own code. A Dice computed here that disagrees with it
    # means `measure` is reading the channels or the cases wrongly, and every millimetre
    # beside it would be wrong for the same reason.
    engine_rows = engine.evaluate_cases("skull", split="test", at="source")
    engine_dice = {row["case"]: row["dice"] for row in engine_rows}
    gaps = [
        abs(row["dice"] - engine_dice[row["case"]]) for row in rows if row["case"] in engine_dice
    ]
    print(
        f"  control: this file's Dice against the engine's, {len(gaps)} cases, "
        f"largest gap {max(gaps):.6f}"
    )

    orders = disagreement(rows)
    print(f"\n  Spearman between the two rankings: {orders['spearman']:.3f}")
    best = orders["best_decile_by_dice"]
    print(
        f"  the best decile by Dice: n={best['n']}  dice {best['dice']:.4f}  "
        f"mae {best['mae_mm']:.3f} mm  worst {best['worst_mm']:.3f} mm"
    )
    worst = sorted(rows, key=lambda row: -abs(row["error_mm"]))[:5]
    print("\n  the five worst in millimetres (case, dice, mm, reference):")
    for row in worst:
        print(
            f"    {row['case']:10s} {row['dice']:.4f}  {row['mm']:7.2f}  "
            f"{row['reference']:7.2f}  {row['error_mm']:+7.2f}"
        )

    record.update(
        {
            "configuration": config,
            "minutes": minutes,
            "history": history.records,
            "engine_rows": engine_rows,
            "summary": totals,
            "disagreement": orders,
            "cases": rows,
        }
    )
    (out / "hc18.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    print(f"\nwrote {out / 'hc18.json'}")

    if arguments.figures:
        figure(
            rows,
            perturbations,
            floor["mae_mm"],
            Path(arguments.figures) / "article-hc18-millimetres.png",
        )


if __name__ == "__main__":
    main()
