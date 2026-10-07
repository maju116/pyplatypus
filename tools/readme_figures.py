"""The figures in README.md, made by running the package.

They sit in the most-read file there is, so they should not be screenshots of a session
nobody can find again - which is what the R package's first mask figure was, and the thing
worth not repeating. Nothing here trains: both figures come from published weights, which
is also the claim the README makes beside them.

    python tools/readme_figures.py boxes
    python tools/readme_figures.py masks \
        --dsbowl examples/data/data_science_bowl/stage1_train \
                 examples/data/data_science_bowl/stage1_validation

`masks` needs every Data Science Bowl case the weights were split out of - give one root
holding all 670, or several that together hold them - and `boxes` needs BCCD. Both reach
the network once, for the weights, and neither trains.

Why `masks` insists on all of them: `dsbowl-unet` was measured on a seeded 80/20 split of
the whole set, so its held-out 134 can only be recovered by making that split again. The
134 images in a directory called `stage1_validation` on this machine are a *different*
split made at a different time, and the model card's own worst case is in its `stage1_train`
- so drawing from that directory would print Dice for images the model was trained on.
Checked rather than assumed, and the check stayed: the card names its worst five cases, and
this refuses to draw if they are not in the split it reconstructed.

The files land in `assets/`, which is tracked, and the README refers to them by absolute
URL rather than by path: PyPI renders the long description on its own domain, where a
relative image is a broken image. They are this repository's own figures - borrowing the R
package's was how the Python README came to show pictures drawn by the other half.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

import pyplatypus as pp
from pyplatypus import split_dataset

ASSETS = Path("assets")


def masks(roots: list[Path]) -> None:
    """Four held-out images, the truth, the prediction and where they disagree.

    A figure and not a score, as in the R package's article: the numbers belong in the
    model card, measured over the whole split. The four Dice values printed on the figure
    are there to say which part of the distribution each row is from, and they are the
    card's own numbers because this is the card's own split.
    """
    pool = Path(tempfile.mkdtemp(prefix="platypus-dsbowl-")) / "cases"
    pool.mkdir(parents=True)
    for root in roots:
        for case in sorted(path for path in root.iterdir() if path.is_dir()):
            (pool / case.name).symlink_to(case.resolve())
    print(f"{len(list(pool.iterdir()))} cases from {len(roots)} root(s)")

    # The split the weights were measured on, remade: same fractions, same seed, same whole
    # set. `publish_dsbowl_weights.py` is where those three came from.
    split = split_dataset(pool, pool.parent / "splits", fractions=(0.8, 0.2), seed=1)
    with Path(split["paths"]["validation"]).open() as handle:
        held_out = {row["key"] for row in csv.DictReader(handle)}

    sidecar = json.loads(Path(pp.resolve_weights("dsbowl-unet")).with_suffix(".json").read_text())
    expected = {row["case"] for row in sidecar["worst_cases"]}
    if not expected <= held_out:
        raise SystemExit(
            "the reconstructed split does not contain the cases the model card measured, so "
            "these are not the images `dsbowl-unet` was held out from. Drawing Dice for them "
            "would be leakage wearing the costume of a result."
        )

    spec = pp.from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": str(split["paths"]["train"]),
                "validation_path": str(split["paths"]["validation"]),
                "colormap": [[0, 0, 0], [255, 255, 255]],
                "mode": "config_file",
            },
            # Architecture, blocks and filters are adopted from the weights, which is the
            # point of naming them: a specification that restated them could disagree with
            # the file, and the file would win.
            "models": [
                {
                    "name": "nuclei",
                    "input_shape": [256, 256],
                    "weights": "dsbowl-unet",
                    "fit": False,
                    "metrics": [{"name": "dice", "include_background": False}],
                }
            ],
        }
    )
    engine = pp.build_engine(spec)
    engine.fit()

    cases = engine.evaluate_cases("nuclei")
    scores = np.array([row["dice"] for row in cases])
    print(f"  split: n={len(scores)} mean={scores.mean():.4f} worst={scores.min():.4f}")
    print("  card : n=134 mean=0.9205 worst=0.7240")

    # Four spread across the ranking rather than the four best: a README showing only what a
    # model does well is an advertisement. The best, two from the middle, and the worst - so
    # the figure covers the distribution the card reports rather than its top.
    ranked = sorted(range(len(cases)), key=lambda index: cases[index]["dice"])
    which = [ranked[-1], ranked[len(ranked) // 2], ranked[len(ranked) // 3], ranked[0]]

    dataset = engine.dataset(engine.runs["nuclei"].spec, "validation")
    figure = pp.plot_masks(
        np.stack([dataset[index][0] for index in which]),
        prediction=engine.predict("nuclei", "validation")[which],
        truth=np.stack([dataset[index][1] for index in which]),
        labels=[f"dice {cases[index]['dice']:.3f}" for index in which],
    )
    path = ASSETS / "README-masks.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")
    print("  dice drawn: " + ", ".join(f"{cases[index]['dice']:.3f}" for index in which))


def boxes(bccd: Path) -> None:
    """One test frame, with what `bccd-yolo3` found on it and what was annotated."""
    root = bccd / "BCCD"
    names = [
        name
        for name in (root / "ImageSets" / "Main" / "test.txt").read_text().split()
        if name.strip()
    ]
    # Absolute, because a path in a CSV is resolved against the CSV's own directory - so a
    # relative one here would be looked for beside the temporary file rather than beside the
    # dataset.
    rows = "\n".join(
        f"{(root / 'JPEGImages' / f'{name}.jpg').resolve()},"
        f"{(root / 'Annotations' / f'{name}.xml').resolve()}"
        for name in names
    )
    listing = Path(tempfile.mkdtemp(prefix="platypus-bccd-")) / "test.csv"
    listing.write_text("images,annotations\n" + rows + "\n")

    classes = ["RBC", "WBC", "Platelets"]
    spec = pp.from_dict(
        {
            "task": "object_detection",
            "data": {
                "train_path": str(listing),
                "validation_path": str(listing),
                "classes": classes,
                "mode": "config_file",
            },
            # The anchors come from the weights too, and a specification naming both is
            # refused: weights mean nothing without the anchors they were trained with.
            "models": [
                {
                    "name": "cells",
                    "architecture": "yolo3",
                    "input_shape": [416, 416],
                    "weights": "bccd-yolo3",
                    "fit": False,
                }
            ],
        }
    )
    engine = pp.build_engine(spec)
    engine.fit()

    found = engine.predict("cells", "validation")
    dataset = engine.dataset(engine.runs["cells"].spec, "validation")
    annotations = dataset.annotations()

    # `plot_boxes` is a vertical montage by design, so a README figure is one frame - and
    # the frame is chosen by reading the annotations rather than by eye, because a picture
    # of red cells alone shows a third of what the detector does.
    wanted = [
        index
        for index, annotation in enumerate(annotations)
        if set(classes) <= set(annotation.names)
    ]
    if not wanted:
        raise SystemExit("no test frame holds all three classes")
    show = wanted[0]
    print(f"drawing {Path(dataset.samples[show].images[0]).name} - the first with all three")

    figure = pp.plot_boxes(
        np.stack([pp.read_image(dataset.samples[show].images[0], size=None)]),
        [found[show]],
        labels=[Path(dataset.samples[show].images[0]).name],
    )
    path = ASSETS / "README-boxes.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")

    # Printed so the caption is transcribed rather than counted off the picture.
    drawn = [
        name
        for name, score in zip(found[show]["names"], found[show]["scores"], strict=True)
        if score >= pp.drawing_style()["box_min_score"]
    ]
    print("  drawn at min_score 0.5: " + ", ".join(f"{n} {drawn.count(n)}" for n in classes))
    print(
        "  annotated: "
        + ", ".join(f"{n} {list(annotations[show].names).count(n)}" for n in classes)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="*", default=None, choices=["masks", "boxes", None])
    parser.add_argument(
        "--dsbowl",
        nargs="+",
        default=[
            "examples/data/data_science_bowl/stage1_train",
            "examples/data/data_science_bowl/stage1_validation",
        ],
        help="roots that together hold every case the weights were split out of",
    )
    parser.add_argument("--bccd", default="examples/data/BCCD_Dataset")
    arguments = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    wanted = arguments.which or ["masks", "boxes"]
    if "masks" in wanted:
        masks([Path(root) for root in arguments.dsbowl])
    if "boxes" in wanted:
        boxes(Path(arguments.bccd))


if __name__ == "__main__":
    main()
