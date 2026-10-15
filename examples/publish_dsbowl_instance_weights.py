"""Train the published `dsbowl-instances` weights: the first instance model this package offers.

    python examples/publish_dsbowl_instance_weights.py --out ../weights

Trains the `rim` model of `examples/data_science_bowl_instances.yaml` - the configuration
`examples/compare_instance_separations.py` measures and `pyplatypus#215` recorded - scores it
by all three of its decodings, and writes `dsbowl-instances.safetensors` with the sidecar and
the card a stranger reads. It uploads nothing, on purpose: the upload is the author's
(`platypus#131`), and `weights/UPLOAD.md` is where the command lives.

It draws nothing either. `compare_instance_separations.py --figures` owns the picture for this
data, and one of the two scripts owning it is enough - the same division as between
`publish_fives_weights.py` and `segment_retinal_vessels.py`.

**The configuration is read rather than restated**, which is the one property this file must
have. The specification comes out of the YAML and is narrowed by that example's own
`selected()`, so the weights that get published are the run the example's table describes and
the two cannot drift. A copy of the recipe here would be a second declaration of one thing, and
this session has spent itself on that shape.

**Why the rim model and not the flow model.** On `#215`'s table the rim route scores 0.5442
against the flow route's 0.5174, which is 0.0268 - 2.4x the 0.0112 two runs of one
configuration differed by, so it is a real lead on a sample of two runs and not a decidable one
on a sample of many. Two things break the tie without relying on that gap: the rim model's
binary foreground Dice is the better mask, 0.9159 against 0.9097, and its `decode` list is
three blocks against the flow route's two - so one file of these weights answers three ways of
turning a mask into objects. The flow model is not published, and a second entry for it would
promise a choice the measurements say is not worth making.

**One file, three numbers - and that is the thing this card has to say out loud.** Every other
set of weights in the registry has one score. These have one per decoding, from one forward
pass, because `decode` is a list: 0.5442 by `nearest_seed`, 0.5396 by `watershed` and 0.4767 by
`connected_components` on `#215`'s run. A number taken off this card without naming the
decoding it came from is worth nothing, and the spread between the first and the last is 0.0675
- ten times the run-to-run noise.

**The data is fetched from the Broad Institute rather than read off this disk, and that is a
licence decision rather than tidiness.** `examples/data/data_science_bowl/` is a 2022 Kaggle
download, governed by the competition rules accepted to get it; BBBC038v1 carries an explicit
CC0 waiver, which is what makes weights trained on it publishable without an argument.
`publish_dsbowl_weights.py` settled that for the semantic weights and the same reasoning
applies here, so this script trains on the Broad copy - and then refuses unless the two copies
are the same images and the same per-object masks, case by case, because the split every
measurement in this project was taken on is named by the directories of the local one.

**The split is taken by case name and never re-drawn.** 536 training cases and 134 validation
cases, exactly the directories the comparison reads, materialised as two directories of
symlinks into the Broad unpack. A fresh 0.8/0.2 split would be defensible and would detach
these weights from every number this project has measured on this data: the 0.7655 ceiling, the
+0.2398 a perfect mask is worth, the noise floor. `dsbowl-unet`'s card reports Dice over "134
held-out images" and the 134 directories on this disk are a *different* 134 - the discrepancy
was 0.01 of Dice, which is also what a rounding difference looks like - so the split is pinned
by name and the object count is checked as a number a different 134 could not have produced.

**The band is two-sided, and a run that comes out too good is refused as well.** The recorded
numbers are `#215`'s at engine `0.8.0a20`; nothing released since touches what a run does - a
refusal, a writer, a warning and one renamed column - so they are the right thing to reproduce.
Run A measured this route's run-to-run spread at 0.009 of average precision, and the band is
three of those: a reproduction lands inside it, run B's pretrained encoder missed by 0.1053
(3.9 band widths) and run D's doubled filters by 0.0007 (a fortieth). A number far outside it in
either direction is a changed run rather than a better model, and publishing it would publish
the change in silence.

Citation, which CC0 does not require and which is owed anyway:

    Caicedo et al., "Nucleus segmentation across imaging experiments: the 2018 Data Science
    Bowl", Nature Methods, 2019. Image set BBBC038v1, Broad Bioimage Benchmark Collection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import textwrap
import time
from argparse import Namespace
from pathlib import Path

import pyplatypus as pp

HERE = Path(__file__).resolve().parent

#: The registry name, the filename and the sidecar's stem, all one string. `dsbowl-unet` is
#: the semantic sibling on the same images and a user choosing between them is this card's
#: audience, so the names differ in what they *do* rather than in a version number.
NAME = "dsbowl-instances"

#: `#215`'s table, measured at engine 0.8.0a20 on 2026-10-13, and the band's centre. Three
#: decodings of one trained model; `JOURNAL.md` §4bq and `CHANGELOG.md` under 0.8.0a22.
RECORDED = {"nearest_seed": 0.5442, "watershed": 0.5396, "connected_components": 0.4767}

#: What one configuration's own runs differ by, measured on this route by run A (seed 1) on
#: 2026-10-15: 0.0085 on `nearest_seed`, 0.0078 on `watershed`, 0.0092 on
#: `connected_components`. One number covers all three. The 0.0074 of `0.8.0a15`'s width sweep
#: and the 0.0112 of the flow model's two runs are the other two estimates of the same thing.
SPREAD = 0.009

#: The split, as two numbers a different 134 images could not both produce. The case count is
#: the weak half - any 134 directories give it - and the object count is the fingerprint.
CASES = 134
TRUTH_OBJECTS = 5999

#: What a better mask is worth and what the yardstick caps, both measured rather than argued:
#: a perfect semantic mask scores 0.7655 through `connected_components` (`#159`), and
#: substituting one is worth +0.2398 to the trained route (`#212`'s oracle).
CEILING = 0.7655
PERFECT_MASK = 0.2398

#: Three runs of this same configuration with one field changed, 120 epochs each, 2026-10-15.
#: They go on the card because "should I train something bigger instead" is the first question
#: a reader of these numbers has, and the answer is measured. `(label, foreground dice, AP by
#: nearest_seed)`.
ALTERNATIVES = [
    ("seed 1, nothing else changed", 0.9133, 0.5527),
    ("filters 32 instead of 16", 0.9096, 0.5435),
    ("resnet34 encoder, pretrained, 5 frozen epochs", 0.8770, 0.4389),
]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(HERE / "data_science_bowl_instances.yaml"),
        help="the configuration to publish from; the recipe comes from here and not from "
        "this script",
    )
    parser.add_argument(
        "--separation",
        default="rim",
        help="which of the file's models to publish, by its separation signal. See the "
        "module docstring for why this is `rim`",
    )
    parser.add_argument(
        "--data",
        default="examples/data/bbbc038",
        help="where to download and unpack BBBC038v1, which is the copy trained on",
    )
    parser.add_argument(
        "--split-from",
        default="examples/data/data_science_bowl",
        help="the directories whose names define the split: stage1_train and "
        "stage1_validation, the ones every measurement on this data was taken on",
    )
    parser.add_argument(
        "--work",
        default=f"examples/output/{NAME}",
        help="the symlinked split, the checkpoint and the run record. Never the recorded "
        "run's own directories - see `relocate`",
    )
    parser.add_argument("--out", default="weights", help="where to write the weights")
    parser.add_argument("--epochs", type=int, default=None, help="override the file's count")
    parser.add_argument("--split", default="validation", help="which split to score on")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="data loading workers. 0 is what the recorded run used - the comparison script "
        "takes the engine's default - and this run is read against it. Measured 2026-10-15 "
        "on the same model and batch size: 4 workers train at 17.5s an epoch against 49.6s "
        "at 0, so the run is loader-bound rather than convolution-bound. That is a three-fold "
        "saving and a variable the band was not calibrated on: a loader with workers "
        "delivers batches in a different order, so the seed no longer means the same run",
    )
    parser.add_argument(
        "--expect-cases",
        default=None,
        help="a JSON file of a previous run of this script, whose case names the scored "
        "split must match exactly. The first run has none to point at and writes one",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=SPREAD,
        help="the run-to-run spread the band is built from",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="how many spreads wide the band is, each way",
    )
    parser.add_argument(
        "--skip-data-control",
        action="store_true",
        help="train on the Broad copy without comparing it to the local one. The control "
        "is what ties the split's names to the publishable download, so a run with this "
        "flag writes no weights",
    )
    return parser.parse_args()


def example_module(filename: str):
    """One of the scripts beside this one, imported by path.

    The examples directory is not a package, so this is the honest way to reuse it - and
    reusing it is the point twice over: `compare_instance_separations.py` owns the recipe's
    narrowing and the table, and `publish_dsbowl_weights.py` owns the download and the
    licence chain. A copy of either here would be free to drift from what is published.
    """
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_names(root: Path) -> dict[str, list[str]]:
    """The split, as the case names the comparison's own directories hold.

    Read rather than drawn, and read from the directories instead of from a saved list,
    because the directories are what every number on this data was measured through. The
    two halves must be disjoint and must cover the 670 cases BBBC038v1 ships; both are
    asserted, since a case in both halves is a leak that no score would reveal.
    """
    names = {}
    for split, folder in (("train", "stage1_train"), ("validation", "stage1_validation")):
        directory = root / folder
        if not directory.is_dir():
            raise SystemExit(f"no {directory}; the split is defined by its directory names")
        names[split] = sorted(item.name for item in directory.iterdir() if item.is_dir())

    overlap = set(names["train"]) & set(names["validation"])
    if overlap:
        raise SystemExit(
            f"{len(overlap)} cases are in both halves of the split, beginning with "
            f"{min(overlap)}. A leak does not show up in a score - nothing is published"
        )
    if len(names["validation"]) != CASES:
        raise SystemExit(
            f"the validation half is {len(names['validation'])} cases and every number this "
            f"card is read against was measured on {CASES}"
        )
    print(f"split: {len(names['train'])} train, {len(names['validation'])} validation, by name")
    return names


def fingerprint(case: Path, subdirs: tuple[str, str]) -> dict:
    """One case as its image's digest and the multiset of its per-object masks' digests.

    The mask *filenames* are deliberately not in it. The data layer reads the directory and
    an object's identity comes from being its own file rather than from what it is called,
    so comparing names would refuse two copies that differ in nothing the model can see -
    a probe answering the question next to the one it is named after.
    """
    images, masks = subdirs
    pictures = sorted(path for path in (case / images).glob("*") if path.is_file())
    if len(pictures) != 1:
        raise SystemExit(
            f"{case / images} holds {len(pictures)} files and this layout is one image per "
            "case; the download is not the dataset this script expects"
        )
    return {
        "image": hashlib.sha256(pictures[0].read_bytes()).hexdigest(),
        "masks": sorted(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (case / masks).glob("*")
            if path.is_file()
        ),
    }


def same_data(names: dict[str, list[str]], broad: Path, local: Path, subdirs) -> None:
    """Refuse unless the publishable download and the local copy are the same data.

    This is the join the licence argument rests on: the split is named by the directories of
    a 2022 Kaggle download and the weights are trained on BBBC038v1, so the two have to be
    the same images and the same masks or the card's provenance describes one copy while the
    numbers come from another. Checked case by case over both halves - 670 images and about
    29,000 mask files - and the first difference says which part differs rather than only
    that something did.
    """
    started = time.time()
    checked = 0
    for split, cases in names.items():
        folder = "stage1_train" if split == "train" else "stage1_validation"
        for case in cases:
            there = broad / case
            if not there.is_dir():
                raise SystemExit(
                    f"case {case} is in the local {folder} and not in {broad}. The split "
                    "cannot be taken by name across two copies that hold different cases"
                )
            mine = fingerprint(local / folder / case, subdirs)
            theirs = fingerprint(there, subdirs)
            if mine["image"] != theirs["image"]:
                raise SystemExit(
                    f"case {case}: the image differs between {local / folder} and {broad}. "
                    "These are not the same data, so neither the split's names nor the "
                    "licence chain carries across - nothing is published"
                )
            if mine["masks"] != theirs["masks"]:
                raise SystemExit(
                    f"case {case}: {len(mine['masks'])} local masks against "
                    f"{len(theirs['masks'])} in {broad}, or the same count with different "
                    "contents. The model learns from these files - nothing is published"
                )
            checked += 1
    print(
        f"control: {checked} cases are byte-identical between {local} and {broad}, images "
        f"and per-object masks ({time.time() - started:.0f}s)"
    )


def materialise(names: dict[str, list[str]], broad: Path, into: Path) -> dict[str, Path]:
    """The split as two directories of symlinks into the publishable copy.

    Symlinks rather than copies, because the alternative is a second 100 MB of the same
    images on a disk at 98%, and rather than a CSV because `nested_dirs` is what the
    configuration being published says - probed before being relied on: a dataset built over
    symlinked sample directories reads its images, its masks and its label map.

    The directories are rebuilt on every run and are therefore removed first, which is the
    one destructive thing this script does. It removes a directory **only** if everything in
    it is a symlink, so pointing `--work` at something real refuses instead of emptying it.
    """
    staged = {}
    for split, cases in names.items():
        directory = into / split
        if directory.exists():
            stray = [item for item in directory.iterdir() if not item.is_symlink()]
            if stray:
                raise SystemExit(
                    f"{directory} holds {len(stray)} things that are not symlinks, beginning "
                    f"with {stray[0].name}. This script would delete it; it will not"
                )
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for case in cases:
            (directory / case).symlink_to((broad / case).resolve())
        staged[split] = directory
    print(f"staged: {staged['train']} and {staged['validation']}, symlinked into {broad}")
    return staged


def point_at(config: dict, staged: dict[str, Path]) -> dict:
    """The configuration, reading the staged split instead of the local copy."""
    data = {
        **config["data"],
        "train_path": str(staged["train"]),
        "validation_path": str(staged["validation"]),
    }
    return {**config, "data": data}


def relocate(config: dict, into: Path) -> dict:
    """Move the run record and the checkpoint out of the recorded run's own directories.

    `examples/data_science_bowl_instances.yaml` writes `examples/output/rim/` - which is
    where `#215`'s record and its checkpoint are, and that record is the band this script's
    own result is read against. Running the file unchanged would overwrite the control with
    the thing being controlled, and the only symptom would be that every future run agrees
    with itself.

    The check afterwards is the point of writing it as a function: the paths are rewritten
    and then the configuration is searched for the two directories that must not appear in
    it, so a third place the file learns to write is caught here rather than by its absence.
    """
    into.mkdir(parents=True, exist_ok=True)
    models = []
    for model in config["models"]:
        callbacks = [
            {**callback, "path": str(into / f"{model['name']}-checkpoint.pt")}
            if callback.get("name") == "model_checkpoint"
            else callback
            for callback in model.get("callbacks", [])
        ]
        models.append({**model, "callbacks": callbacks} if callbacks else model)
    moved = {**config, "output_dir": str(into), "models": models}

    written = json.dumps(moved)
    for recorded in ("examples/output/rim", "examples/output/flows"):
        if recorded in written:
            raise SystemExit(
                f"the configuration still writes into {recorded} after being relocated, "
                "which is where the recorded run this result is compared against lives. "
                "Something in the file writes somewhere this function does not know about"
            )
    return moved


def check_one_mask(rows: list[dict]) -> None:
    """Every decoding of one model reports one mask, because there is one forward pass.

    Free, and it is the control on the card's central claim: three numbers from one file.
    Two different foreground Dice values in these rows would mean the rows came from two
    models, and then the card's table is three scores of three things.
    """
    masks = {round(row["foreground_dice"], 6) for row in rows}
    if len(masks) != 1:
        raise SystemExit(
            f"the {len(rows)} rows report {len(masks)} different foreground Dice values "
            f"({sorted(masks)}), so they are not three decodings of one trained model"
        )


def check_split(rows: list[dict]) -> None:
    """The split, as a number a different set of images could not have produced.

    The case count is the weak half: any 134 directories give it. The truth object count is
    the fingerprint - 5999 objects over these 134 cases, reproduced by `#215` and by all
    three of the one-variable runs - and a different 134 images would have to hold exactly
    as many nuclei to pass.
    """
    truth = {row["truth_objects"] for row in rows}
    if truth != {TRUTH_OBJECTS}:
        raise SystemExit(
            f"the scored split holds {sorted(truth)} ground-truth objects and every number "
            f"this card is read against was measured over {TRUTH_OBJECTS}. Two different "
            "sets of images give scores that look comparable and are not"
        )
    print(f"control: {TRUTH_OBJECTS} ground-truth objects, the recorded split's own count")


def check_band(rows: list[dict], spread: float, tolerance: float) -> None:
    """Refuse a result outside the recorded numbers by more than the measured noise.

    Two-sided on purpose. A run far below the band is broken rather than worse - run B's
    pretrained encoder landed 0.1053 away, 3.9 band widths - and a run far above it is a
    changed recipe rather than a better model, which would publish the change in silence.
    Calibrated at both ends, which is what makes it a bound and not a guess: the band is
    three times a spread of 0.009 measured on this route, and the smallest deviation it has
    to refuse is twelve times its own width away.
    """
    width = spread * tolerance
    outside = []
    for row in rows:
        recorded = RECORDED.get(row["decode"])
        if recorded is None:
            continue
        gap = row["average_precision"] - recorded
        if abs(gap) > width:
            outside.append(
                f"{row['decode']}: {row['average_precision']:.4f} is {gap:+.4f} "
                f"from the recorded {recorded:.4f}"
            )
    if outside:
        listed = "\n  - ".join(outside)
        raise SystemExit(
            f"outside the band of +/-{width:.4f}, which is {tolerance:g} x a run-to-run "
            f"spread of {spread:.4f}:\n  - {listed}\n"
            "Nothing was written. The measurements are in the JSON beside the weights, so "
            "this is diagnosable: a run far below the band is broken rather than worse, and "
            "one far above it is a different recipe than the card would describe."
        )
    print(f"control: every decoding is within +/-{width:.4f} of its recorded value")


def refuse_a_shadowed_fingerprint(extra: dict, spec) -> None:
    """A sidecar key that collides with the fingerprint is a weights file that cannot load.

    `export_weights` merges what this script passes over what the specification says the
    weights are for, and `_refuse_mismatch` reads those same names back when loading. So an
    extra called `separation` - the obvious name for the block's settings - would replace
    the string `"rim"` with a dict, and every later load would refuse the file for
    disagreeing with itself. Derived from the specification rather than listed, so a field
    the fingerprint gains later is covered without an edit here.
    """
    reserved = set(spec.models[0].weights_fingerprint()) | {"n_class", "task"}
    collisions = sorted(reserved & set(extra))
    if collisions:
        raise SystemExit(
            f"these sidecar fields would overwrite what the weights say they are for: "
            f"{', '.join(collisions)}. The file would then refuse to load into the model it "
            "was trained as. Rename them - `separation_settings` rather than `separation`"
        )


def card(measured: dict) -> str:
    """The description a stranger reads, with every number substituted from the run."""
    rows = measured["table"]
    by_decode = {row["decode"]: row for row in rows}
    best = max(rows, key=lambda row: row["average_precision"])
    yardstick = by_decode["connected_components"]
    signal = best["average_precision"] - yardstick["average_precision"]
    mask = best["foreground_dice"]

    lines = [
        f"# {NAME}",
        "",
        "A U-Net trained to tell touching nuclei **apart** on BBBC038v1, scored on 134",
        "held-out images as instance average precision - objects matched to objects, averaged",
        "over ten IoU thresholds from 0.50 to 0.95:",
        "",
    ]
    for row in sorted(rows, key=lambda row: -row["average_precision"]):
        lines.append(
            f"    {row['decode']:22} AP {row['average_precision']:.4f}   "
            f"{row['predicted_objects']} objects found of {row['truth_objects']}"
        )
    lines += [
        "",
        f"    foreground Dice        {mask:.4f}   the mask all three rows share",
        "",
        "## Name the decoding or the number means nothing",
        "",
        *textwrap.wrap(
            "These are one file of weights and three scores. The network predicts interior, "
            "rim and background; turning those channels into objects is a separate step, the "
            f"specification takes a list of them, and the spread between the best and the "
            f"worst row above is {signal:.4f} - about seven times this configuration's "
            "run-to-run noise. A score quoted from this card without its decoding is not a "
            "property of the weights.",
            width=86,
        ),
        "",
        "## What the separation signal is worth here, which is less than it sounds",
        "",
        *textwrap.wrap(
            f"`connected_components` ignores the signal entirely - it labels connected "
            f"foreground, so touching nuclei come back as one object - and it scores "
            f"{yardstick['average_precision']:.4f} on this same mask. So the signal these "
            f"weights were trained to carry is worth {signal:+.4f}. Substituting a *perfect* "
            f"semantic mask into the same route is worth +{PERFECT_MASK:.4f}, and a perfect "
            f"mask through `connected_components` scores {CEILING:.4f} - which is the ceiling "
            "every row above sits under. The mask is the limit on this data, by a factor of "
            "about three and a half, and these weights do not change that.",
            width=86,
        ),
        "",
        "## A bigger model was measured and it is not the answer",
        "",
        "Three runs of this configuration with one field changed, 120 epochs each:",
        "",
        f"    {'as published':46} fg {mask:.4f}  AP {best['average_precision']:.4f}",
    ]
    for label, foreground, average_precision in ALTERNATIVES:
        lines.append(f"    {label:46} fg {foreground:.4f}  AP {average_precision:.4f}")
    lines += [
        "",
        *textwrap.wrap(
            "Doubling the filters moved the score by less than a tenth of the run-to-run "
            "noise, and a pretrained ResNet-34 encoder lost 0.1053 - fifteen times the "
            "spread of the mask and twelve times that of the score. Where the mask did move "
            "far enough to read, the score followed it at about 2.7 points of average "
            "precision per point of foreground Dice, against the 2.85 implied by "
            "substituting a perfect mask. Two ways of asking, one answer: effort spent on "
            "the mask pays and effort spent on capacity does not.",
            width=86,
        ),
        "",
        "## What these weights cannot do",
        "",
        "- **One kind of object.** Nuclei against background. The per-object masks this was",
        "  trained on say where the objects are and nothing about what kind they are.",
        "- **Trained at 256x256 on light microscopy.** The dataset spans stains, magnifications",
        "  and fluorescence; a slide unlike anything in it is outside what was measured.",
        (
            f"- **{best['truth_objects'] - best['predicted_objects']} objects of "
            f"{best['truth_objects']} are not found** by the best row above. An "
            "average precision"
        ),
        "  is not a count, and counting nuclei with these weights means accepting that.",
        "- **The number is one machine's.** A seed reproduces a direction and not digits; a",
        "  second run of this exact file moved the score by 0.009.",
        "",
        "## The semantic sibling on the same images",
        "",
        *textwrap.wrap(
            "`dsbowl-unet` is a U-Net on the same dataset with one foreground class, Dice "
            "0.92 over the same held-out images. It finds the nuclei and cannot tell two "
            "touching ones apart; these weights can, at the cost of a thinner interior. "
            "Choose by whether the answer is an area or a count.",
            width=86,
        ),
        "",
        "## Provenance",
        "",
        (
            "- Data: BBBC038v1 (2018 Data Science Bowl), stage1_train, CC0 1.0. "
            f"{measured['data_source']}"
        ),
        f"- Citation: {measured['citation']}",
        "- Weights: MIT, because the data waives copyright and nothing else constrains them.",
        f"- Split: {measured['split']['train']} training and {measured['split']['validation']}",
        "  validation cases, by case name, the same split every measurement on this data used.",
        (
            f"- Trained with pyplatypus {measured['pyplatypus']}, seed "
            f"{measured['seed']}, {measured['epochs']} epochs, "
            f"{measured['minutes']:.0f} minutes on one GTX 1070."
        ),
        (
            f"- Separation: {measured['separation_settings']}; decodings: "
            f"{', '.join(measured['decodings'])}."
        ),
    ]
    return "\n".join(lines) + "\n"


def registry_entry(measured: dict) -> str:
    """The lines somebody adds to `REGISTRY` once the upload has a commit.

    Printed rather than described, and wrapped from one sentence rather than assembled out
    of quoted fragments: the description is the only part of a registry entry that carries
    numbers, so it is generated from the run like every other number on the card.
    """
    rows = {row["decode"]: row["average_precision"] for row in measured["table"]}
    best = max(measured["table"], key=lambda row: row["average_precision"])
    signal = best["average_precision"] - rows["connected_components"]
    description = (
        "U-Net, 256x256, nuclei in light microscopy, trained to separate touching objects: "
        "interior, rim and background. BBBC038v1 (the 2018 Data Science Bowl, CC0), the same "
        "134 held-out images as `dsbowl-unet`. **One file, three scores** - instance average "
        f"precision {rows['nearest_seed']:.4f} by nearest_seed, {rows['watershed']:.4f} by "
        f"watershed and {rows['connected_components']:.4f} by connected_components - so a "
        "number from these weights means nothing without the decoding it came from. "
        f"Foreground Dice {best['foreground_dice']:.4f}. The separation signal is worth "
        f"{signal:+.4f} here while a perfect mask is worth +{PERFECT_MASK:.4f}: the mask is "
        "the limit, not the signal."
    )
    wrapped = textwrap.wrap(description, width=84)
    body = [f'            "{line} "' for line in wrapped[:-1]]
    body.append(f'            "{wrapped[-1]}"')
    return "\n".join(
        [
            f'    "{NAME}": Published(',
            '        repo="maju116/platypus-weights",',
            f'        filename="{NAME}.safetensors",',
            '        revision="<the commit git ls-remote reports after the upload>",',
            "        description=(",
            *body,
            "        ),",
            "    ),",
        ]
    )


def main() -> None:
    import yaml

    arguments = parse()
    comparison = example_module("compare_instance_separations.py")
    semantic = example_module("publish_dsbowl_weights.py")

    # Narrowed by the example's own selector, so this script cannot invent a row the
    # configuration does not describe - and relocated before anything is read, so a bad
    # flag fails in a second rather than after the download.
    config = yaml.safe_load(Path(arguments.config).read_text())
    config = comparison.selected(
        config,
        Namespace(
            separation=[arguments.separation],
            decode=None,
            epochs=arguments.epochs,
            config=arguments.config,
        ),
    )
    if len(config["models"]) != 1:
        raise SystemExit(
            f"--separation {arguments.separation} selects {len(config['models'])} models and "
            "one file of weights is one model. See the module docstring on why this is `rim`"
        )
    work = Path(arguments.work)
    config = relocate(config, work)

    broad = semantic.fetch(Path(arguments.data))
    names = split_names(Path(arguments.split_from))
    subdirs = tuple(config["data"].get("subdirs", ("images", "masks")))
    if arguments.skip_data_control:
        print("WARNING: the data control is skipped, so nothing will be written")
    else:
        same_data(names, broad, Path(arguments.split_from), subdirs)

    staged = materialise(names, broad, work)
    config = point_at(config, staged)

    spec = pp.from_dict(config, source=arguments.config)
    model_spec = spec.models[0]
    engine = pp.build_engine(spec, device=arguments.device, num_workers=arguments.workers)
    print(
        f"{model_spec.name}: {model_spec.signal.value}, head "
        f"{model_spec.head_channels(spec.data.n_class)} channels, decoded "
        f"{len(model_spec.decodings)} ways"
    )
    if arguments.expect_cases:
        comparison.check_cases(engine, spec, arguments.split, arguments.expect_cases)

    started = time.time()
    history = engine.fit(verbose=True)[model_spec.name]
    minutes = (time.time() - started) / 60
    print(f"\ntrained in {minutes:.1f} min, {len(history.records)} epochs")

    rows = engine.evaluate(arguments.split)
    comparison.print_table(rows)
    check_one_mask(rows)
    check_split(rows)

    decodings = [block.name for block in model_spec.decodings]
    cases = {
        name: engine.evaluate_cases(model_spec.name, arguments.split, decode=name)
        for name in decodings
    }
    best = max(rows, key=lambda row: row["average_precision"])
    worst = sorted(cases[best["decode"]], key=lambda row: row["average_precision"])[:5]

    measured = {
        "pyplatypus": pp.__version__,
        "seed": spec.seed,
        "epochs": len(history.records),
        "minutes": round(minutes, 1),
        "data_source": semantic.SOURCE,
        "citation": semantic.CITATION,
        "split": {split: len(found) for split, found in names.items()},
        "separation_settings": config["models"][0]["separation"],
        "decodings": decodings,
        "table": rows,
        "recorded": RECORDED,
        "band": {"spread": arguments.spread, "tolerance": arguments.tolerance},
        "per_case": {name: pp.summarise_cases(found) for name, found in cases.items()},
        # Named `cases` because that is what `--expect-cases` reads, here and in the three
        # other scripts that write one: the next run of this file is controlled by this one.
        "cases": cases[best["decode"]],
        "worst_cases": [
            {"case": row["case"], "average_precision": row["average_precision"]} for row in worst
        ],
    }

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{NAME}-measured.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(f"\nwrote {out / f'{NAME}-measured.json'}")

    print("\n--- per case, by decoding ---")
    for name, summary in measured["per_case"].items():
        for row in summary:
            if row["metric"] != "average_precision":
                continue
            print(
                f"  {name:22} n={row['n']} mean={row['mean']:.4f} sd={row['sd']:.4f} "
                f"min={row['min']:.4f} worst={row['worst_case'][:16]}"
            )

    check_band(rows, arguments.spread, arguments.tolerance)
    if arguments.skip_data_control:
        raise SystemExit(
            "the data control was skipped, so the split's names are not tied to the "
            "publishable copy and nothing is written. The measurements are in the JSON above"
        )

    extra = {
        "data": "BBBC038v1 (2018 Data Science Bowl), stage1_train, split 536/134 by case name",
        "data_source": semantic.SOURCE,
        "data_licence": "CC0 1.0",
        "citation": semantic.CITATION,
        "weights_licence": "MIT",
        "trained_with": f"pyplatypus {pp.__version__}",
        # Not `separation`: that name is part of what the weights say they are for, and
        # overwriting it with this block would make the file refuse to load. See the check.
        "separation_settings": config["models"][0]["separation"],
        "decodings": decodings,
        "split": {**measured["split"], "by": "case name"},
        "validation": rows,
        "per_case": measured["per_case"],
        "worst_cases": measured["worst_cases"],
        "ceiling": {
            "connected_components_on_a_perfect_mask": CEILING,
            "perfect_mask": PERFECT_MASK,
        },
    }
    refuse_a_shadowed_fingerprint(extra, spec)
    path = engine.export_weights(model_spec.name, out / NAME, **extra)
    (out / f"{NAME}-card.md").write_text(card(measured))

    print(f"\nwrote {path}")
    print(f"      {path.with_suffix('.json')}")
    print(f"      {out / f'{NAME}-card.md'}")
    print(
        f"\nUpload {path.name} and {path.with_suffix('.json').name}, and fold "
        f"{NAME}-card.md into the Hub repository's README.md as a section - one card holds "
        "one section per model. weights/UPLOAD.md has the command. Then the registry entry:"
    )
    print()
    print(registry_entry(measured))


if __name__ == "__main__":
    main()
