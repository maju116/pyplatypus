"""What a split by slice costs, measured against the same split by patient.

    python examples/measure_split_leak.py --data examples/data/msd-spleen/Task09_Spleen

A volume dataset sliced into 2D images has many samples per patient, and neighbouring
slices of one abdomen are nearly the same picture. Split those samples at random and the
model is validated on slices whose neighbours it trained on: the reported Dice is partly a
memory test, and nothing in the number says so. `split_dataset(group_by=)` prevents it, and
until now nothing had measured what it prevents.

**The lie is not the gap between two runs.** The first version of this measured a
slice-split run against a patient-split run and reported the difference. That comparison is
rigged in a way nothing in it says: the two arms validate on **different patients**, so the
gap mixes the leak with how hard that particular validation set happens to be. Measured
before this file was rewritten - three patient draws of the honest arm alone span 0.0392 of
Dice, against a "leak" of 0.0431, and the same split trained twice moves 0.026. The effect
and the noise were the same size and the first design could not tell them apart.

**So the lie is measured on one model instead.** A fifth of the patients are held out and
neither arm ever sees them. Each arm then reports the validation Dice a user would quote, and is
scored again on those twelve:

    the size of the lie  =  what an arm says about itself  -  what it achieves on unseen patients

That subtraction is one model on two sets, so the choice of patients and the training noise
cancel rather than being compared across runs. The honest arm's own version of the same
subtraction is the control: it should be small, and if it is not, this measurement is about
something else.

## The naming, which this example states rather than assumes

`split_dataset` discovers `nested_dirs`: one directory per sample, holding `images/` and
`masks/`. So a slice is a directory and the patient has to be recoverable from its name -
here `patient_XX_slice_YYY`, and `group_by=r"^(patient_\\d+)_"` is what reads it. Written
down because a pattern that matches nothing is the failure this example is about: it would
be refused rather than silently falling back to one group per slice, and that refusal is
the mechanism, not a detail.

## The data

MSD Task09 (Spleen), **CC BY-SA 4.0** - read from `license.txt` in the dataset's own store
and from `dataset.json` inside the archive rather than from a registry page. 41 labelled CT
volumes, sliced axially. Only slices where the spleen is present are kept: an abdomen is
mostly not spleen, and a dataset of empty masks would measure the background class instead
of the thing. That choice is stated because it changes what the number means.

No weights are published from this. The artefact is a subtraction, and a model trained on a
leaked split is the last thing to put on a hub.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import pyplatypus as pp

# Hounsfield units: the window a radiologist reads an abdomen at. Fixed rather than taken
# from each volume's own extremes, so the same tissue is the same grey in every slice.
SOFT_TISSUE = (50.0, 400.0)


def slice_volumes(root: Path, out: Path, minimum: int) -> dict:
    """One directory per slice, named so the patient is recoverable from it."""
    import nibabel as nib
    from PIL import Image

    images = sorted((root / "imagesTr").glob("spleen_*.nii.gz"))
    if not images:
        raise SystemExit(f"no volumes under {root / 'imagesTr'}; see the module docstring")

    centre, width = SOFT_TISSUE
    low, high = centre - width / 2, centre + width / 2
    kept, patients = 0, 0
    for index, image_path in enumerate(images):
        label_path = root / "labelsTr" / image_path.name
        if not label_path.exists():
            continue
        volume = nib.as_closest_canonical(nib.load(str(image_path))).get_fdata()
        labels = nib.as_closest_canonical(nib.load(str(label_path))).get_fdata()
        patients += 1
        for z in range(volume.shape[2]):
            mask = (labels[:, :, z] > 0).astype(np.uint8)
            if int(mask.sum()) < minimum:
                continue
            picture = np.clip((volume[:, :, z] - low) / (high - low), 0, 1)
            case = out / f"patient_{index:02d}_slice_{z:03d}"
            (case / "images").mkdir(parents=True, exist_ok=True)
            (case / "masks").mkdir(parents=True, exist_ok=True)
            # Into the subdirectories, not beside them. The first version of this wrote
            # both files into the case root with `images/` and `masks/` created and empty,
            # and `discover_samples` refused all 1043 as incomplete - which is the reader
            # doing its job, and the reason this comment is here rather than a fixed line.
            Image.fromarray((picture * 255).astype(np.uint8), mode="L").save(
                case / "images" / "ct.png"
            )
            Image.fromarray(mask * 255, mode="L").save(case / "masks" / "spleen.png")
            kept += 1
    print(f"  {patients} patients, {kept} slices kept (>= {minimum} spleen pixels)")
    return {"patients": patients, "slices": kept}


def specification(split_dir: Path, size: int, epochs: int, batch: int) -> dict:
    return {
        "task": "semantic_segmentation",
        "data": {
            "train_path": str(split_dir / "train.csv"),
            "validation_path": str(split_dir / "validation.csv"),
            "test_path": str(split_dir / "test.csv"),
            "mode": "config_file",
            "colormap": [[0, 0, 0], [255, 255, 255]],
        },
        "models": [
            {
                "name": "spleen",
                "architecture": "u_net",
                "input_shape": [size, size],
                "channels": 1,
                "blocks": 4,
                "filters": 16,
                "batch_size": batch,
                "epochs": epochs,
                "loss": {"name": "cce_dice"},
                "metrics": [{"name": "dice", "include_background": False}],
                "optimizer": {"name": "adam", "learning_rate": 0.001},
            }
        ],
        "seed": 1,
    }


def rows_in(path: Path) -> int:
    return len(path.read_text().strip().split("\n")) - 1


def read_rows(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text().strip().split("\n")
    return lines[0], lines[1:]


def patients_in(path: Path) -> set[str]:
    found = set()
    for row in read_rows(path)[1]:
        match = re.match(r"^(patient_\d+)_", row.split(",")[0])
        if match:
            found.add(match.group(1))
    return found


def leaky_split(honest: Path, where: Path, seed: int) -> None:
    """The mistake, built by hand, because the package will not make it.

    `split_dataset(group_by=)` is the thing under test and it refuses to produce this. So
    the leaking arm is assembled here from the honest arm's own development rows - the same
    29 patients and the same slices - shuffled at the level of a slice rather than a
    patient. The test rows are copied across untouched, so both arms are judged on exactly
    the same twelve held-out patients.
    """
    import random

    where.mkdir(parents=True, exist_ok=True)
    header, rows = read_rows(honest / "train.csv")
    rows = rows + read_rows(honest / "validation.csv")[1]
    random.Random(seed).shuffle(rows)
    cut = round(0.8 * len(rows))
    (where / "train.csv").write_text("\n".join([header, *rows[:cut]]) + "\n")
    (where / "validation.csv").write_text("\n".join([header, *rows[cut:]]) + "\n")
    (where / "test.csv").write_text((honest / "test.csv").read_text())


def run(where: Path, arguments) -> dict:
    spec = pp.from_dict(specification(where, arguments.size, arguments.epochs, arguments.batch))
    engine = pp.build_engine(spec, num_workers=arguments.workers)
    engine.fit(verbose=False)
    reported = engine.evaluate(split="validation")[0]
    held_out = engine.evaluate(split="test")[0]
    return {
        "reported": float(reported["dice"]),
        "held_out": float(held_out["dice"]),
        "overstatement": float(reported["dice"]) - float(held_out["dice"]),
        "train_slices": rows_in(where / "train.csv"),
        "validation_slices": rows_in(where / "validation.csv"),
        "shared_patients": len(
            patients_in(where / "train.csv") & patients_in(where / "validation.csv")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/msd-spleen/Task09_Spleen")
    parser.add_argument("--out", default="measurements/spleen")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum", type=int, default=100, help="spleen pixels per slice")
    parser.add_argument("--slices", default=None, help="a directory of slices already cut")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    slices = Path(arguments.slices) if arguments.slices else out / "slices"
    if arguments.slices is None:
        print("--- slicing ---")
        shape = slice_volumes(Path(arguments.data), slices, arguments.minimum)
    else:
        shape = {"patients": None, "slices": len(list(slices.iterdir()))}
        print(f"--- reusing {shape['slices']} slices from {slices} ---")

    # Patient-disjoint three ways first. The test third is the twelve neither arm sees, and
    # it is built by the function under test rather than by hand, because an honest holdout
    # is exactly what `group_by` is for.
    honest = out / "honest"
    # 0.6/0.2/0.2 rather than 0.7/0.1/0.2, and the reason is the control rather than taste.
    # At a tenth of 41 patients the honest arm validates on **three** of them, so its own
    # self-overstatement - the number that says whether the leaking arm's is noise - rested
    # on three abdomens and moved 0.0135 between two runs of the identical configuration
    # while the leaking arm's moved 0.0033. Measured that way first, which is why this
    # comment exists instead of a tidier fraction.
    pp.split_dataset(
        slices,
        honest,
        fractions=(0.6, 0.2, 0.2),
        seed=1,
        group_by=r"^(patient_\d+)_",
    )
    leaky = out / "leaky"
    leaky_split(honest, leaky, seed=1)

    held = patients_in(honest / "test.csv")
    print("\n--- the held-out patients, seen by neither arm ---")
    print(f"  {len(held)} patients, {rows_in(honest / 'test.csv')} slices")
    for label, where in (("leaky", leaky), ("honest", honest)):
        development = patients_in(where / "train.csv") | patients_in(where / "validation.csv")
        assert not (development & held), f"{label} arm touches a held-out patient"
    print("  control: neither arm's development set touches them")

    results = {}
    for label, where in (("split by slice", leaky), ("split by patient", honest)):
        print(f"\n--- {label} ---")
        results[label] = run(where, arguments)
        row = results[label]
        print(f"  patients on both sides of its own split: {row['shared_patients']}")
        print(
            f"  reports {row['reported']:.4f}   achieves {row['held_out']:.4f}   "
            f"overstates by {row['overstatement']:+.4f}"
        )

    leaked, straight = results["split by slice"], results["split by patient"]
    print("\n--- the size of the lie ---")
    print(f"  a slice-split run overstates itself by {leaked['overstatement']:+.4f}")
    print(
        f"  a patient-split run overstates itself by {straight['overstatement']:+.4f}  "
        f"<- the control"
    )
    print(
        f"  and on the same unseen patients the two models differ by "
        f"{leaked['held_out'] - straight['held_out']:+.4f}"
    )
    print(f"\n  {straight['held_out']:.4f} is what this package achieves on this data.")

    (out / "split-leak.json").write_text(
        json.dumps({"data": shape, "held_out_patients": len(held), "runs": results}, indent=2)
        + "\n"
    )
    print(f"  wrote {out / 'split-leak.json'}")


if __name__ == "__main__":
    main()
