"""Train the published `fives-unet` weights, from the dataset's own train/test split.

    python examples/publish_fives_weights.py --out weights/

Trains the retinal-vessel model that `examples/segment_retinal_vessels.py` documents, scores
it per case and per disease, joins those scores to the quality grading FIVES ships, and writes
`fives-unet.safetensors` with a sidecar recording all of it plus a card a stranger can read.
It uploads nothing, on purpose: the upload is the author's. It draws nothing either -
`segment_retinal_vessels.py --figures` is where the worst cases get a picture, and one of
those two scripts owning the figure is enough.

**The configuration is imported rather than copied.** Both halves of this - the split writer
and the specification - come from `segment_retinal_vessels.py`, so the weights that get
published are the run that example describes and the two cannot drift. 40 epochs at a 512
tile, which is the example's own default and about 6.7 hours on a GTX 1070; the vignette's
15-epoch run is the right trade for a document that has to rebuild and the wrong one for
weights that outlive it.

**What the card has to say, and the reason this script computes it.** The per-disease table
reads image quality as pathology. FIVES ships `Quality Assessment.xlsx` grading every image
for illumination and contrast, blur, and low contrast; nothing in the pipeline uses it, and
joined to the per-case scores it inverts the finding - **14 of the 15 low-contrast images in
the test split are glaucoma**, and on the images graded clean glaucoma is no longer the worst
group. A card reporting `glaucoma 0.84` beside `normal 0.90` would be actively misleading, so
the numbers that go on it are generated here rather than typed by somebody reading a table.

**The licence, which is not the same as the other two.** `dsbowl-unet` was trained on CC0 data
and `bccd-yolo3` on MIT, so neither constrained anything. FIVES is **CC BY 4.0**, weights are
plausibly a derivative work, and attribution therefore has to travel with them: the sidecar
carries `data_licence`, the citation and the source, and `weights_licence` is CC BY 4.0 rather
than MIT. That is the conservative reading and the author's to overrule.

Citation, which CC BY 4.0 does require:

    Jin et al., "FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel
    Segmentation", Scientific Data 9, 475 (2022). figshare 19688169, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

import pyplatypus as pp

CITATION = (
    "Jin et al., FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel "
    "Segmentation, Scientific Data 9, 475 (2022). figshare 19688169."
)
SOURCE = "https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169"
SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
#: The grading sheet's second worksheet is the test split, and 1 means *good*: 185 of the 200
#: test images carry 1 under LC and the 15 that do not are the low-contrast ones. Asserted
#: below rather than trusted, because reading the polarity backwards would invert every claim
#: the card makes.
TEST_SHEET = "xl/worksheets/sheet2.xml"
GRADES = {"C": "illumination_contrast", "D": "blur", "E": "low_contrast"}


def example_module():
    """`segment_retinal_vessels.py`, imported by path.

    The examples directory is not a package, so this is the honest way to reuse it - and
    reusing it is the point: a copy of `specification()` here would let the published weights
    drift from the run the example documents, which is the one thing this script must not do.
    """
    path = Path(__file__).with_name("segment_retinal_vessels.py")
    spec = importlib.util.spec_from_file_location("segment_retinal_vessels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quality_grades(root: Path) -> dict[str, dict[str, int]]:
    """FIVES' own quality grading for the test split, read without a spreadsheet library.

    An `.xlsx` is a zip of XML, and this needs three columns of 200 rows - not a reason to
    add pandas and openpyxl to an example. Keys are built as `number_disease`, which is how
    the image files are named and therefore how a case is keyed everywhere else here.
    """
    workbook = root / example_module().FOLDER / "Quality Assessment.xlsx"
    if not workbook.exists():
        raise SystemExit(f"no quality grading at {workbook}; it ships with the dataset")

    with zipfile.ZipFile(workbook) as archive:
        shared = [
            "".join(node.text or "" for node in item.iter(f"{SHEET}t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(f"{SHEET}si")
        ]
        rows = []
        for row in ET.fromstring(archive.read(TEST_SHEET)).iter(f"{SHEET}row"):
            cells = {}
            for cell in row.findall(f"{SHEET}c"):
                column = "".join(character for character in cell.get("r") if character.isalpha())
                value = cell.find(f"{SHEET}v")
                text = value.text if value is not None else None
                if cell.get("t") == "s" and text is not None:
                    text = shared[int(text)]
                cells[column] = text
            rows.append(cells)

    graded = {}
    for cells in rows[1:]:
        if not cells.get("A") or not cells.get("B"):
            continue
        graded[f"{cells['B']}_{cells['A']}"] = {
            name: int(cells.get(column, "1") or "1") for column, name in GRADES.items()
        }
    return graded


def check_polarity(graded: dict[str, dict[str, int]]) -> None:
    """1 is good, 0 is flagged - and reading it the other way round inverts the card.

    The dataset's paper does not say so in a sentence this script can quote, so the polarity
    is established from the data: 15 of 200 test images are flagged for low contrast and 14 of
    those 15 are glaucoma. Under the opposite reading it would be 185 images, which is not a
    flag, and the glaucoma share would be 36 of 185 - indistinguishable from chance.
    """
    flagged = [key for key, grades in graded.items() if grades["low_contrast"] == 0]
    glaucoma = [key for key in flagged if key.endswith("_G")]
    if not (len(graded) == 200 and len(flagged) == 15 and len(glaucoma) == 14):
        raise SystemExit(
            "the quality grading does not look the way every claim on this card depends on: "
            f"{len(graded)} graded, {len(flagged)} flagged for low contrast, "
            f"{len(glaucoma)} of those glaucoma, expected 200 / 15 / 14. Either the sheet "
            "changed or the polarity is the other way round - do not publish until this is "
            "understood."
        )


def by_quality(cases: list[dict], graded: dict[str, dict[str, int]], column: str) -> dict:
    """The same metric over all cases, over the ones graded clean, and over the rest."""

    def summarise(rows: list[dict]) -> dict:
        values = [row[column] for row in rows]
        return {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if values else None,
        }

    def clean(row: dict) -> bool:
        grades = graded.get(row["case"], {})
        return all(grades.get(name, 1) == 1 for name in GRADES.values())

    return {
        "all": summarise(cases),
        "graded_clean": summarise([row for row in cases if clean(row)]),
        "graded_poor": summarise([row for row in cases if not clean(row)]),
    }


def by_disease_and_quality(cases: list[dict], graded: dict, column: str) -> dict:
    """Per disease, over the clean images only - which is the table that inverts the finding."""
    module = example_module()
    out: dict[str, list[float]] = {}
    for row in cases:
        grades = graded.get(row["case"], {})
        if not all(grades.get(name, 1) == 1 for name in GRADES.values()):
            continue
        found = re.search(module.BY_DISEASE, row["case"])
        name = module.GROUPS[found.group(1)] if found else "?"
        out.setdefault(name, []).append(row[column])
    return {
        name: {
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for name, values in sorted(out.items())
    }


def card(measured: dict) -> str:
    """The description a stranger reads, with every number substituted from the run."""
    overall = measured["per_case"]["dice"]
    disease = measured["by_disease"]["dice"]
    clean = measured["by_disease_clean"]["dice"]
    quality = measured["by_quality"]["dice"]
    worst_disease = min(disease, key=lambda name: disease[name]["mean"])
    worst_clean = min(clean, key=lambda name: clean[name]["mean"])

    lines = [
        "# fives-unet",
        "",
        "A U-Net trained on FIVES, scored on the dataset's own 200-image test split:",
        "",
        (
            f"    Dice      mean {overall['mean']:.4f}  sd {overall['sd']:.4f}  "
            f"worst {overall['min']:.4f}"
        ),
        (
            f"    clDice    mean {measured['per_case']['cldice']['mean']:.4f}  "
            f"sd {measured['per_case']['cldice']['sd']:.4f}"
        ),
        "",
        "## Read the per-disease table with the quality grading beside it",
        "",
        "FIVES is 200 test images, 50 of each of normal, AMD, diabetic retinopathy and",
        "glaucoma, and the per-disease means look like a finding about disease:",
        "",
    ]
    for name, row in disease.items():
        lines.append(f"    {name:22} n={row['n']:3}  dice {row['mean']:.4f}  sd {row['sd']:.4f}")
    lines += [
        "",
        "**It is not.** The dataset ships a quality grading that nothing in the pipeline",
        "uses, and 14 of the 15 test images flagged for low contrast are glaucoma. Over the",
        "images graded clean on all three axes the table reads differently:",
        "",
    ]
    for name, row in clean.items():
        lines.append(f"    {name:22} n={row['n']:3}  dice {row['mean']:.4f}  sd {row['sd']:.4f}")
    lines += [
        "",
        (
            f"    graded clean   n={quality['graded_clean']['n']:3}  "
            f"dice {quality['graded_clean']['mean']:.4f}"
        ),
        (
            f"    graded poor    n={quality['graded_poor']['n']:3}  "
            f"dice {quality['graded_poor']['mean']:.4f}"
        ),
        "",
        # Wrapped rather than typed: two group names are substituted into this sentence, so
        # where the lines break depends on the run. Hand-wrapping it left a three-word line
        # in the first card this script produced.
        *textwrap.wrap(
            f"So `{worst_disease}` is the weakest group over all images and `{worst_clean}` "
            "over the clean ones. **A per-disease number taken off this card is a statement "
            "about picture quality as much as about pathology**, and a clinical conclusion "
            "drawn from it would be wrong.",
            width=86,
        ),
        "",
        "## What else these weights cannot promise",
        "",
        "- **Vessels, not lesions.** One foreground class, the vessel tree.",
        "- **The number is one machine's.** A seed gives identical results to twelve decimal",
        "  places on a processor and differs in the fourth on a card, so the published Dice is",
        "  reproducible in direction and not in digits. `trained_with` records the engine",
        "  version, and a change to how data is drawn changes what a seed draws.",
        "- **Trained at a 512 tile over the 2048 source.** Predicting a resized 512 image",
        "  instead gives vessels about a quarter the width the model was trained on.",
        "",
        "## Provenance",
        "",
        f"- Data: FIVES, CC BY 4.0. {SOURCE}",
        f"- Citation: {CITATION}",
        "- Weights: CC BY 4.0, because they are a derivative of CC BY data and attribution",
        "  travels with them.",
        (
            f"- Trained with pyplatypus {pp.__version__}, seed {measured['seed']}, "
            f"{measured['epochs_run']} epochs of {measured['epochs']}."
        ),
    ]
    return "\n".join(lines) + "\n"


def with_checkpoint(configuration: dict, out: Path) -> dict:
    """The example's configuration plus a checkpoint, which is the one deviation from it.

    It changes no arithmetic. Six and a half hours unattended with nothing on disk until the
    end means a crash costs the night; `save_best_only` writes on improvement and touches
    nothing else, and the sidecar records the specification that trained these weights -
    including this callback, so a reader sees it rather than having to trust that it did
    nothing.

    **It watches what the example's `early_stopping` watches, and that is not a detail.**
    `restore_best` hands back the epoch the stopping rule judged best, so a checkpoint
    monitoring anything else would leave a file on disk from one epoch while the weights that
    get published come from another - two things named for the same run and measured on
    different ones. A test holds the two monitors equal rather than this paragraph.
    """
    configuration["models"][0]["callbacks"].append(
        {
            "name": "model_checkpoint",
            "path": str(out / "fives-unet-checkpoint.safetensors"),
            "monitor": "val_dice",
            "save_best_only": True,
        }
    )
    return configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/fives")
    parser.add_argument("--out", default="weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--loss", default="cce_dice")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.80,
        help="refuse to write the weights if the mean Dice comes in below this",
    )
    arguments = parser.parse_args()

    module = example_module()
    root = Path(arguments.data)
    graded = quality_grades(root)
    check_polarity(graded)
    print(f"quality grading: {len(graded)} test images, polarity checked")

    splits = module.write_splits(root, root / "splits")
    configuration = module.specification(splits, arguments)

    configuration = with_checkpoint(configuration, Path(arguments.out))
    Path(arguments.out).mkdir(parents=True, exist_ok=True)
    spec = pp.from_dict(configuration)

    started = time.time()
    engine = pp.Engine(spec, device=arguments.device, num_workers=arguments.workers)
    history = engine.fit(verbose=True)["vessels"]
    minutes = (time.time() - started) / 60
    print(f"\ntrained in {minutes:.1f} min, {len(history.records)} epochs")

    print("\n--- the split as one number ---")
    for row in engine.evaluate():
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    cases = engine.evaluate_cases("vessels")
    distribution = {row["metric"]: row for row in pp.summarise_cases(cases)}
    measured = {
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "epochs_run": len(history.records),
        "minutes": round(minutes, 1),
        "per_case": distribution,
        "by_disease": {
            metric: module.by_group(cases, metric) for metric in ("dice", "iou", "cldice")
        },
        "by_disease_clean": {
            metric: by_disease_and_quality(cases, graded, metric)
            for metric in ("dice", "iou", "cldice")
        },
        "by_quality": {
            metric: by_quality(cases, graded, metric) for metric in ("dice", "iou", "cldice")
        },
    }

    print("\n--- per case ---")
    for metric, row in distribution.items():
        print(
            f"  {metric}: n={row['n']} mean={row['mean']:.4f} sd={row['sd']:.4f} "
            f"min={row['min']:.4f} worst={row['worst_case']}"
        )
    print("\n--- per disease, all images / graded clean ---")
    for name in measured["by_disease"]["dice"]:
        everything = measured["by_disease"]["dice"][name]
        clean = measured["by_disease_clean"]["dice"].get(name, {})
        print(
            f"  {name:22} all n={everything['n']:3} dice={everything['mean']:.4f} "
            f"sd={everything['sd']:.4f} | clean n={clean.get('n', 0):3} "
            f"dice={clean.get('mean', float('nan')):.4f}"
        )

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fives-unet-measured.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(f"\nwrote {out / 'fives-unet-measured.json'}")

    # The refusal `publish_dsbowl_weights.py` taught: weights whose measurement did not come
    # out are not published, because the card is the measurement. A run that lands here below
    # the floor has still written its numbers above, so the failure is diagnosable rather
    # than silent.
    dice = distribution["dice"]["mean"]
    if dice < arguments.floor:
        raise SystemExit(
            f"mean Dice {dice:.4f} is below the floor of {arguments.floor:.2f}, so nothing "
            "was written. The previous runs of this configuration measured 0.8509 at 40 "
            "epochs and 0.8245 at 15; a number far below that is a broken run rather than a "
            "worse model, and publishing it would publish the breakage."
        )

    worst = sorted(cases, key=lambda row: row["dice"])[:5]
    path = engine.export_weights(
        "vessels",
        out / "fives-unet",
        data="FIVES, the authors' own train/test split",
        data_source=SOURCE,
        data_licence="CC BY 4.0",
        citation=CITATION,
        weights_licence="CC BY 4.0",
        trained_with=f"pyplatypus {pp.__version__}",
        split={"train": 600, "test": 200, "seed": arguments.seed},
        validation=list(distribution.values()),
        by_disease=measured["by_disease"]["dice"],
        by_disease_graded_clean=measured["by_disease_clean"]["dice"],
        by_quality=measured["by_quality"]["dice"],
        worst_cases=[{"case": row["case"], "dice": row["dice"]} for row in worst],
    )
    (out / "fives-unet-card.md").write_text(card(measured))

    print(f"\nwrote {path}")
    print(f"      {path.with_suffix('.json')}")
    print(f"      {out / 'fives-unet-card.md'}")
    # Two files, not three. On the Hub one card holds one section per model - which is how
    # the two already-published weights are written - so the card here is the generated source
    # for a section of that `README.md` rather than a file of its own.
    print(
        f"\nUpload {(out / 'fives-unet.safetensors').name} and {(out / 'fives-unet.json').name}, "
        f"and fold {(out / 'fives-unet-card.md').name} into the Hub repository's README.md as "
        "a section. Then add the commit to REGISTRY in pyplatypus/weights.py."
    )


if __name__ == "__main__":
    main()
