"""What a permuted `channels_from` looks like: a plausible answer and no error.

    python examples/measure_channel_order.py --data examples/data/msd-braintumour/Task01_BrainTumour

`channels_from`'s own description ends with a claim and no evidence: *"A model trained with
FLAIR in channel one and used on data where channel one is T1 returns a plausible answer and
no error."* A sentence like that needs a test or it should not be written, and the honest
test for this one is a picture - a number would be a number about one permutation of
twenty-four.

## MSD repackaged BraTS, so the layout this field is for has to be rebuilt

`channels_from` is for datasets keeping one modality per file, which is how BraTS ships.
MSD Task01 stacks all four into one 4D file - measured, `(240, 240, 155, 4)`, one file per
case - so there is nothing to order. The first step writes the four out separately, and
**the order comes from `dataset.json`'s own `modality` map** rather than from sorting:

    {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}

That is not a workaround. It is what anyone holding MSD data must do before this field
applies, and writing it on the page is where the ground truth about which channel is which
becomes visible. Sorting the filenames gives `flair, t1, t1gd, t2`, which is reproducible and
anatomically meaningless - the same trap one level down.

## The permutation is a second specification, not a flag

The engine should not gain an argument for this: permuting channels is a mistake, not a
feature. So the wrong run is the same trained weights read back through a specification whose
four patterns are named in a different order, which is exactly how a person does it by
accident - by copying a configuration and editing the list.

## What is deliberately not measured

Every permutation. Twenty-four orderings of four modalities would give twenty-four Dice
scores and no understanding; one picture of one wrong order shows the thing the sentence
claims. The score is reported beside it only to make the point that **it is not zero**.

## The data

MSD Task01 (BrainTumour), **CC BY-SA 4.0** - from `license.txt` in the dataset's own store
and `dataset.json` inside the archive. Labels are `{0, 1, 2, 3}`: background, oedema,
non-enhancing tumour, enhancing tumour. Every volume is already 1 x 1 x 1 mm at
240 x 240 x 155, so `target_spacing` is a no-op here and the crop is well defined.

No weights are published: the artefact is the pair of pictures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import pyplatypus as pp

# From dataset.json, not from sorting. The whole example rests on this being the truth.
MODALITIES = ("flair", "t1", "t1gd", "t2")
# One swap, not a shuffle: FLAIR and T1 change places and the other two stay. A reader can
# see which half of the picture to blame.
SWAPPED = ("t1", "flair", "t1gd", "t2")


def patterns_for(order: tuple[str, ...]) -> list[str]:
    """The `channels_from` patterns naming four modality files in a given order."""
    return [rf"_{name}\.nii" for name in order]


def _rgb(colour: str) -> tuple[int, int, int]:
    """A `#rrggbb` string as three 0-255 numbers."""
    return (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16))


IN_ORDER = patterns_for(MODALITIES)
PERMUTED = patterns_for(SWAPPED)
# Background, then the three tumour classes in the engine's own class palette rather than in
# three colours invented here: `style.py` owns that decision and R reads the same list.
COLORMAP = [(0, 0, 0), *(_rgb(colour) for colour in pp.drawing_style()["class_colours"][:3])]


def split_modalities(root: Path, out: Path, cases: int) -> list[str]:
    """One file per modality, which is the layout `channels_from` is for."""
    import nibabel as nib

    images = sorted((root / "imagesTr").glob("BRATS_*.nii.gz"))[:cases]
    keys = []
    for path in images:
        label_path = root / "labelsTr" / path.name
        if not label_path.exists():
            continue
        key = path.name.replace(".nii.gz", "")
        case = out / key
        if (case / "masks" / "seg.nii.gz").exists():
            keys.append(key)
            continue
        (case / "images").mkdir(parents=True, exist_ok=True)
        (case / "masks").mkdir(parents=True, exist_ok=True)
        image = nib.load(str(path))
        stack = np.asarray(image.dataobj, dtype=np.float32)
        for index, name in enumerate(MODALITIES):
            nib.save(
                nib.Nifti1Image(stack[..., index], image.affine),
                str(case / "images" / f"{key}_{name}.nii.gz"),
            )
        label = nib.load(str(label_path))
        nib.save(
            nib.Nifti1Image(np.asarray(label.dataobj, dtype=np.int16), label.affine),
            str(case / "masks" / "seg.nii.gz"),
        )
        keys.append(key)
    print(f"  {len(keys)} cases, {len(MODALITIES)} modalities each -> {out}")
    return keys


def specification(splits, patterns, size, epochs, checkpoint, fit=True):
    model = {
        "name": "tumour",
        "architecture": "u_net",
        "input_shape": [size, size, size],
        "channels": len(MODALITIES),
        "blocks": 3,
        "filters": 16,
        "batch_size": 1,
        "epochs": epochs,
        "loss": {"name": "cce_dice"},
        "metrics": [{"name": "dice", "include_background": False}],
        "optimizer": {"name": "adam", "learning_rate": 0.001},
    }
    if fit:
        model["callbacks"] = [
            {"name": "model_checkpoint", "monitor": "val_dice", "path": str(checkpoint)}
        ]
    else:
        model["fit"] = False
        model["weights"] = str(checkpoint)
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(splits["train"]),
            "validation_path": str(splits["validation"]),
            "mode": "config_file",
            "labels": [0, 1, 2, 3],
            "window": "auto",
            "channels_from": patterns,
        },
        "models": [model],
        "seed": 1,
    }


def write_splits(cases_root: Path, out: Path) -> dict[str, Path]:
    """Through `split_dataset`, in `nested_dirs`, which is the mode that reads several files.

    Not a hand-written CSV: a `config_file` row names one image file per sample, and the
    whole point here is four. `nested_dirs` discovers everything in a case's `images/`, which
    is what gives `channels_from` a list of files to match its patterns against.
    """
    found = pp.split_dataset(cases_root, out, fractions=(0.8, 0.2), seed=1)
    written = {name: out / f"{name}.csv" for name in ("train", "validation")}
    sizes = {name: len(path.read_text().strip().split("\n")) - 1 for name, path in written.items()}
    print("  " + ", ".join(f"{n} {s}" for n, s in sizes.items()) + f", skipped {found['skipped']}")
    return written


def one_case_split(validation: Path, case: str, out: Path) -> Path:
    """The same split with one row in it, which is what makes the picture unambiguous.

    `predict(space="source")` returns a list in the order the samples were discovered and
    nothing in the return value says which case an entry is. A split holding one case
    answers that by construction instead of by trusting two orderings to agree.
    """
    header, *rows = validation.read_text().strip().split("\n")
    chosen = [row for row in rows if row.split(",")[0] == case]
    if len(chosen) != 1:
        raise pp.PlatypusError(f"{len(chosen)} rows for {case} in {validation}, wanted one")
    out.write_text("\n".join([header, *chosen]) + "\n")
    return out


def as_picture(volume: np.ndarray) -> np.ndarray:
    """One volume scaled to 0-1 by its own 1st and 99th percentile.

    A drawing decision rather than the model's: what the network saw is whatever
    `window: auto` gave it. An MRI file holds raw intensities, and `plot_masks` scales a
    float image only when its maximum is at most 1 - so drawn unscaled, every voxel above 1
    clips to white and the anatomy disappears.
    """
    low, high = np.percentile(volume, (1, 99))
    return np.clip((volume - low) / max(high - low, 1e-6), 0.0, 1.0)


def figure(splits, case, expected, out, size, checkpoint, workers) -> None:
    """Two rows: one set of weights reading one scan through two orders of four files.

    The image column shows **channel one of what that row was given** - FLAIR where the
    order is right and T1 where it is not - because the mistake is in what arrived under
    that name, and drawing the same picture twice would hide the only thing that differs.

    `expected` is the Dice the full split measured for this case in the right order, and the
    right-order row has to reproduce it: that is the control on the one-row split naming the
    case the median was chosen from, which nothing else here would catch.
    """
    import nibabel as nib

    one_case = one_case_split(splits["validation"], case, out / "split" / "figure.csv")
    case_root = out / "cases" / case
    truth = np.asarray(nib.load(str(case_root / "masks" / "seg.nii.gz")).dataobj)
    # The plane holding the most tumour, so the picture is not an argument about which slice
    # was chosen. Any class above 0: background is everywhere and would choose nothing.
    plane = int(np.argmax((truth > 0).sum(axis=(0, 1))))

    images, predictions, labels = [], [], []
    for order in (MODALITIES, SWAPPED):
        shown = pp.build_engine(
            pp.from_dict(
                specification(
                    {"train": splits["train"], "validation": one_case},
                    patterns_for(order),
                    size,
                    1,
                    checkpoint,
                    fit=False,
                )
            ),
            num_workers=workers,
        )
        shown.fit(verbose=False)
        dice = shown.evaluate_cases()[0]["dice"]
        if order == MODALITIES and abs(dice - expected) > 1e-4:
            raise pp.PlatypusError(
                f"the one-row split scores {dice:.4f} where the full split scored "
                f"{expected:.4f} for {case}: the picture is not of the case it names"
            )
        first = case_root / "images" / f"{case}_{order[0]}.nii.gz"
        images.append(as_picture(np.asarray(nib.load(str(first)).dataobj, dtype=np.float32)))
        predictions.append(shown.predict(split="validation", space="source")[0])
        labels.append(f"channel one is {order[0].upper()}   dice {dice:.4f}")
        print(f"  {order[0]} first: dice {dice:.4f}")

    drawn = pp.plot_masks(
        np.stack(images)[..., None],
        prediction=np.stack(predictions),
        truth=np.stack([truth, truth]),
        colormap=COLORMAP,
        labels=labels,
        slice=plane,
    )
    path = out / "channel-order.png"
    drawn.savefig(str(path), dpi=140)
    print(f"  slice {plane} of {truth.shape[-1]}, wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/msd-braintumour/Task01_BrainTumour")
    parser.add_argument("--out", default="measurements/braintumour")
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="draw the picture from the checkpoint already on disk, without training again",
    )
    arguments = parser.parse_args()

    out = Path(arguments.out)
    # `.pt`, not `.safetensors`: a checkpoint is the run's safety net rather than a
    # publication, so it is a torch archive on purpose. It was a workaround when this was
    # written - the callback wrote one format whatever the path said and the reader chose
    # the other from the extension - which is `#256`, found here and fixed in `0.8.0a31`.
    checkpoint = out / "tumour.pt"
    print("--- one file per modality, the layout this field is for ---")
    split_modalities(Path(arguments.data), out / "cases", arguments.cases)
    splits = write_splits(out / "cases", out / "split")

    if not arguments.figure_only:
        print("\n--- trained with the order dataset.json states ---")
        engine = pp.build_engine(
            pp.from_dict(
                specification(splits, IN_ORDER, arguments.size, arguments.epochs, checkpoint)
            ),
            num_workers=arguments.workers,
        )
        engine.fit(verbose=False)
        right = engine.evaluate()[0]
        print(f"  dice {right['dice']:.4f}")

        print("\n--- the same weights, read through a permuted channels_from ---")
        print(f"  patterns: {PERMUTED}")
        wrong_engine = pp.build_engine(
            pp.from_dict(
                specification(
                    splits, PERMUTED, arguments.size, arguments.epochs, checkpoint, fit=False
                )
            ),
            num_workers=arguments.workers,
        )
        wrong_engine.fit(verbose=False)
        wrong = wrong_engine.evaluate()[0]
        print(f"  dice {wrong['dice']:.4f}   and no error was raised")

        print("\n--- the claim in `channels_from`'s own description ---")
        print(f"  right order  dice {right['dice']:.4f}")
        print(
            f"  FLAIR and T1 swapped  dice {wrong['dice']:.4f}  "
            f"({wrong['dice'] / right['dice'] * 100:.0f}% of it, not zero)"
        )

        (out / "channel-order.json").write_text(
            json.dumps(
                {
                    "right": right,
                    "permuted": wrong,
                    "patterns": {"right": IN_ORDER, "wrong": PERMUTED},
                },
                indent=2,
                default=float,
            )
            + "\n"
        )
        print(f"  wrote {out / 'channel-order.json'}")

    print("\n--- the case to draw: the median of the split, not the best or the worst ---")
    scored = pp.build_engine(
        pp.from_dict(specification(splits, IN_ORDER, arguments.size, 1, checkpoint, fit=False)),
        num_workers=arguments.workers,
    )
    scored.fit(verbose=False)
    rows = sorted(scored.evaluate_cases(), key=lambda row: row["dice"])
    median = rows[len(rows) // 2]
    print(f"  {median['case']}  dice {median['dice']:.4f}, the median of {len(rows)}")
    figure(
        splits,
        median["case"],
        median["dice"],
        out,
        arguments.size,
        checkpoint,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
