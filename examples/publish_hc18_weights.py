"""Train the published `hc18-unet` weights, and publish a card whose headline is not a Dice.

    python examples/publish_hc18_weights.py --out weights/

Trains the fetal-head model that `examples/measure_fetal_head_circumference.py` documents,
scores it per case in **both** units, and writes `hc18-unet.safetensors` with a sidecar
recording all of it plus a card a stranger can read. It uploads nothing, on purpose: the
upload is the author's. It draws nothing either - `measure_fetal_head_circumference.py
--figures` owns the figure, and one of the two scripts owning it is enough.

**The configuration is imported rather than copied.** `specification()`, `prepare()` and
`measure()` all come from that example, so the weights published here are the run it
describes and the two cannot drift. `tests/test_examples.py` holds the defaults of the two
scripts equal for every argument that shapes the model, because a copy of `--filters` here
would be a published model the example does not document.

## Why this card is shaped differently from the other four

Every other set of weights in this registry reports an overlap score, because for every
other one that *is* the question. Here it is not. A head circumference is what a clinician
writes down, and the measurement this model exists for is in millimetres - so the card leads
with millimetres, carries the Dice beside them, and reports how badly the two orderings
disagree.

That disagreement is the finding, and it is why a Dice alone would be a misleading card.
Measured on three runs of this one recipe the two columns rank the same cases with a
Spearman between 0.15 and 0.62 - see `RUNS` - and **tens of the cases scoring 0.95 or
better are still wrong by more than 2 mm** in every run that counted them. A reader who
picked these weights off a Dice would be picking them on the wrong number. The range is
quoted rather than one run's figure because this script used to say "about 0.6" here while
publishing a run that measured 0.15: the direction is the finding and the magnitude is one
draw.

**The millimetres are reported twice, against two different references, and both belong on
the card.** `mae_mm` is against what the sonographer recorded, which is what somebody acting
on the number would see; it carries about +0.7 mm that belongs to the dataset rather than to
the model, because the annotation is a drawn curve of finite width and the masks are that
curve filled. `model_mae_mm` is against the circumference of the truth mask - the same
arithmetic on both sides, and therefore the model's own contribution. Reporting only the
first blames the model for a drawing convention; reporting only the second hides the
convention from the person reading millimetres.

**The floor is on millimetres too.** A run that trains badly is refused here on the number
the weights are for, not on the overlap, which is the same decision the card is about.

## The licence

HC18 is **CC BY 4.0**, read from the dataset's own Zenodo record (`10.5281/zenodo.1327317`:
`license` `cc-by-4.0`, `access_right` `open`, 176,488,085 bytes across four files, published
2018-07-27) rather than from a mirror or an aggregator. So these weights are the **same
bracket as `fives-unet`** rather than a new one - CC BY 4.0 rather than MIT, because the data
is and weights are plausibly a derivative work. The sidecar carries `data_licence`, the
citation and the source, and `weights_licence` is CC BY 4.0. That is the conservative reading
and the author's to overrule.

The challenge page names no licence at all and asks for the paper and the dataset to be
cited, which is the attribution CC BY 4.0 already requires.

Citation, which CC BY 4.0 does require:

    van den Heuvel, de Bruijn, de Korte and van Ginneken, "Automated measurement of fetal
    head circumference using 2D ultrasound images", PLoS ONE 13(8), e0200412 (2018).
    Zenodo 10.5281/zenodo.1327317, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

import pyplatypus as pp

CITATION = (
    "van den Heuvel, de Bruijn, de Korte and van Ginneken, Automated measurement of fetal "
    "head circumference using 2D ultrasound images, PLoS ONE 13(8), e0200412 (2018). "
    "Zenodo 10.5281/zenodo.1327317."
)
SOURCE = "https://zenodo.org/records/1327317"
#: The Dice at or above which a case counts as one a reader would have published. The card's
#: central claim is a count of those that are still wrong in millimetres, so the threshold is
#: named here rather than buried in a comprehension.
PUBLISHABLE_DICE = 0.95
#: And the millimetre error above which a measurement is wrong enough to matter clinically.
#: Chosen as a round number a reader can re-apply rather than derived, and the card says so.
MATTERS_MM = 2.0

#: The same recipe run three times on 2026-10-25, `(label, mean foreground Dice, Spearman,
#: the share of the publishable cases still over `MATTERS_MM`)`. `None` where a run did not
#: compute that share - the first predates the field.
#:
#: They are on the card because the disagreement is its central claim and one draw of it is
#: not a measurement. All three are far from 1, so the direction is the finding; the
#: magnitude moves by a factor of four between them, so a card quoting a single figure as a
#: property of the task would be overstating it fourfold at one end. Note which way it
#: moves: the better the Dice, the *less* it orders anything, which is what a statistic
#: computed over a narrower spread of a better model does. Three runs is a direction and not
#: a law.
RUNS = [
    ("the run `measure_fetal_head_circumference.py` measures", 0.9347, 0.620, None),
    ("a second run, nothing changed, refused by the floor", 0.9418, 0.435, 63 / 145),
    ("the run published here", 0.9494, 0.153, 85 / 144),
]


def example_module():
    """`measure_fetal_head_circumference.py`, imported by path.

    The examples directory is not a package, so this is the honest way to reuse it - and
    reusing it is the point: a copy of `specification()` here would let the published weights
    drift from the run the example documents, which is the one thing this script must not do.
    """
    path = Path(__file__).with_name("measure_fetal_head_circumference.py")
    spec = importlib.util.spec_from_file_location("measure_fetal_head_circumference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def with_checkpoint(configuration: dict, out: Path) -> dict:
    """The example's configuration plus a checkpoint, which is the one deviation from it.

    It changes no arithmetic. `save_best_only` writes on improvement and touches nothing
    else, and the sidecar records the specification that trained these weights - including
    this callback, so a reader sees it rather than having to trust that it did nothing.

    **It watches what the example's `early_stopping` watches, and that is not a detail.**
    `restore_best` hands back the epoch the stopping rule judged best, so a checkpoint
    monitoring anything else would leave a file on disk from one epoch while the weights that
    get published come from another - two things named for the same run and measured on
    different ones. A test holds the two monitors equal rather than this paragraph.
    """
    stopping = next(
        callback
        for callback in configuration["models"][0]["callbacks"]
        if callback["name"] == "early_stopping"
    )
    configuration["models"][0]["callbacks"].append(
        {
            "name": "model_checkpoint",
            "path": str(out / "hc18-unet-checkpoint.safetensors"),
            "monitor": stopping["monitor"],
            "save_best_only": True,
        }
    )
    return configuration


def spread(rows: list[dict]) -> dict:
    """The millimetre error's quantiles, because the mean describes the tail and not a case.

    Measured on two runs of this configuration: mean 5.66 and 5.22 mm against medians of
    2.00 and 2.19, with a worst case near 90. So a card leading with the mean alone reports
    a number no typical case resembles, and one leading with the median alone hides that a
    few cases are out by the width of a head. Both, with the quantiles between them.
    """
    errors = np.abs([row["error_mm"] for row in rows])
    return {
        "median_mm": float(np.median(errors)),
        "p75_mm": float(np.percentile(errors, 75)),
        "p90_mm": float(np.percentile(errors, 90)),
    }


def published_but_wrong(rows: list[dict]) -> dict:
    """How many cases a reader would have accepted on Dice and should not have.

    The card's central claim, computed rather than typed. `n_publishable` is the cases at or
    above `PUBLISHABLE_DICE`; `over` is how many of those are wrong by more than
    `MATTERS_MM`. The worst of them is named, because a count without an example reads as a
    rounding artefact.
    """
    good = [row for row in rows if row["dice"] >= PUBLISHABLE_DICE]
    over = [row for row in good if abs(row["error_mm"]) > MATTERS_MM]
    worst = max(good, key=lambda row: abs(row["error_mm"])) if good else None
    return {
        "dice_at_least": PUBLISHABLE_DICE,
        "matters_mm": MATTERS_MM,
        "n": len(rows),
        "n_publishable": len(good),
        "n_over": len(over),
        "share_over": (len(over) / len(good)) if good else 0.0,
        "worst_case": None if worst is None else worst["case"],
        "worst_dice": None if worst is None else worst["dice"],
        "worst_error_mm": None if worst is None else worst["error_mm"],
    }


def card(measured: dict) -> str:
    """The Hub card's section for these weights, generated from the run rather than typed.

    Generated because the numbers are the point. A card whose figures were copied by hand
    from a terminal is a card that goes stale at the next release and says nothing about it.
    """
    totals = measured["summary"]
    quantiles = measured["spread"]
    floor = measured["pipeline_control"]
    split = measured["split"]
    gap = measured["disagreement"]
    wrong = measured["published_but_wrong"]
    best = gap["best_decile_by_dice"]

    # The claim this card exists to make, and the degenerate case said rather than crashed
    # on: a run where nothing reaches the threshold cannot make it, and a card that quietly
    # printed `None` would be worse than one that says so. The floor in `main` is what
    # catches a *broken* run; this is for a weak one that still passed it.
    if wrong["n_publishable"] == 0:
        claim = (
            f"No test case reached a Dice of {wrong['dice_at_least']:.2f}, so the count this "
            "card would otherwise report cannot be made on this run."
        )
    else:
        claim = (
            f"Of the {wrong['n_publishable']} test cases scoring "
            f"{wrong['dice_at_least']:.2f} or better - a figure anybody would publish - "
            f"**{wrong['n_over']} are still wrong by more than "
            f"{wrong['matters_mm']:.0f} mm** ({wrong['share_over'] * 100:.0f}%), the worst "
            f"of them `{wrong['worst_case']}` at Dice {wrong['worst_dice']:.4f} and "
            f"{wrong['worst_error_mm']:+.2f} mm."
        )

    # And the spread of the claim, which matters as much as the claim: a Spearman is the one
    # figure on this card measured on a single run and quoted as a property. `RUNS` says what
    # three runs of the one recipe gave, and the rows are generated from it so that editing
    # the constant and not the card is impossible.
    rows = "\n".join(
        f"        | {label} | {dice:.4f} | {rho:.2f} | "
        f"{'not counted' if share is None else format(share * 100, '.0f') + '%'} |"
        for label, dice, rho, share in RUNS
    ).lstrip()
    spread_note = textwrap.dedent(f"""
        **And the magnitude of that disagreement is one draw.** The same recipe was run
        {len(RUNS)} times and the Spearman moved from {max(r[2] for r in RUNS):.2f} to
        {min(r[2] for r in RUNS):.2f}. The direction holds in all {len(RUNS)} - every one of
        them is far from 1, and every run that counted the share found tens of publishable
        cases wrong by more than {MATTERS_MM:.0f} mm - so read this section as *the two
        orderings disagree* rather than as any one number.

        | run | mean Dice | Spearman | of the publishable, over {MATTERS_MM:.0f} mm |
        |---|---|---|---|
        {rows}

        Which way it moves is worth the sentence: the better the Dice, the *less* it orders
        anything, which is what a rank statistic does when the thing it ranks has a narrower
        spread. {len(RUNS)} runs is a direction and not a law, and the published row is the
        one this card's figures come from.
        """).strip()
    spread_note = textwrap.indent(spread_note, " " * 8).lstrip()

    return textwrap.dedent(f"""
        ## hc18-unet

        A U-Net that segments the fetal skull in 2D ultrasound, trained on HC18, for
        measuring **head circumference in millimetres**. `{split["train"]}` training images,
        `{split["validation"]}` validation and `{split["test"]}` test, divided **by
        examination rather than by row**: the dataset contains several images of the same
        echoscopic examination, and a row-wise split puts near-duplicates on both sides of
        it.

        The headline here is not an overlap score, and that is deliberate.

        | what | value |
        |---|---|
        | **median** absolute error against the sonographer's measurement | **{quantiles["median_mm"]:.2f} mm** |
        | mean absolute error against it | {totals["mae_mm"]:.2f} mm |
        | 75th and 90th percentile | {quantiles["p75_mm"]:.2f} mm, {quantiles["p90_mm"]:.2f} mm |
        | bias against it | {totals["bias_mm"]:+.2f} mm |
        | worst case | {totals["worst_mm"]:.2f} mm |
        | mean absolute error against the truth mask's own circumference | {totals["model_mae_mm"]:.2f} mm |
        | bias against it | {totals["model_bias_mm"]:+.2f} mm |
        | mean Dice (foreground) | {totals["dice"]:.4f} |
        | worst Dice | {totals["dice_worst"]:.4f} |

        **The median and the mean are both here because they disagree.** The error
        distribution has a long tail - half the cases are inside
        {quantiles["median_mm"]:.2f} mm and a tenth are outside
        {quantiles["p90_mm"]:.2f} mm - so the mean describes the tail rather than a typical
        case, and the median alone would hide that some cases are out by the width of a
        head. Read the worst case as the reason this model is not a measuring instrument
        without a human looking at it.

        **Two references, and both belong here.** The first pair is against what the
        sonographer recorded, which is what somebody acting on the number sees; it carries
        **{floor["bias_mm"]:+.2f} mm** that belongs to the dataset rather than to this
        model. HC18's annotation is a drawn curve {floor["line_width_px"]:.2f} px wide and
        the training masks are that curve filled, so every mask is half a line width
        outside the ellipse the sonographer drew: `pi * w * pixel` predicts
        {floor["predicted_bias_mm"]:+.2f} mm against the {floor["bias_mm"]:+.2f} measured
        by putting the truth masks through this same pipeline, a correlation of
        {floor["correlation"]:.4f} over {floor["n"]} cases with
        {floor["residual_mae_mm"]:.3f} mm left unexplained. The second pair is against the
        circumference of the truth mask - the same arithmetic on both sides - and is
        therefore this model's own contribution.

        ### Do not choose these weights on the Dice

        The two columns rank the same cases differently: Spearman
        **{gap["spearman"]:.2f}** between the ordering by Dice and the ordering by absolute
        millimetre error. {claim} The best decile by Dice averages {best["dice"]:.4f} and
        still has a mean absolute error of {best["mae_mm"]:.2f} mm, worst
        {best["worst_mm"]:.2f} mm.

        {spread_note}

        Why: a ragged boundary barely moves an integral over the region, and a mask that is
        systematically fat or thin moves it linearly. Those are the two errors Dice is
        respectively most and least able to see, and they are the wrong way round for
        anybody measuring a length.

        ### Measuring with them

        Predict on the **source** grid and measure there. The pixel size is a property of
        the file, so a prediction resized to this model's input shape has pixels of a size
        nobody recorded, and multiplying by the file's number would be wrong by the ratio of
        the two shapes.

        ```python
        from pyplatypus import ellipse_circumference

        for case, prediction in engine.predict_stream("skull", split="test", space="source"):
            mask = prediction.argmax(axis=-1) == 1
            mm = ellipse_circumference(mask, spacing=(pixel_mm, pixel_mm))
        ```

        From R, the same thing is `ellipse_measure(mask, spacing = c(px, px))`.

        ### Data and licence

        HC18, **CC BY 4.0** ({SOURCE}). These weights are released under **CC BY 4.0**
        rather than MIT because the data is, and attribution travels with them.

        {CITATION}

        Trained with pyplatypus {measured["trained_with"]}, {measured["epochs"]} epochs,
        {measured["minutes"]:.0f} minutes on one GTX 1070.
        """).strip()


def registry_entry(measured: dict, shape: tuple[int, int]) -> str:
    """The lines somebody adds to `REGISTRY` once the upload has a commit.

    Printed rather than described, for the reason the card is generated: the description is
    the only part of a registry entry that carries numbers, and a number copied out of a
    terminal into a registry is one nothing holds to the run. `UPLOAD.md` said this script
    printed the entry before it did, which is what prompted writing it.

    The entry leads with millimetres and names the Dice only inside the sentence warning the
    reader off it, because that order *is* the finding - `tests/test_weights.py` asserts it.
    """
    totals = measured["summary"]
    median = measured["spread"]["median_mm"]
    wrong = measured["published_but_wrong"]
    description = (
        f"U-Net, {shape[0]}x{shape[1]}, the fetal skull in 2D ultrasound, for measuring "
        "**head circumference in millimetres**. HC18 (CC BY 4.0), split by examination "
        f"rather than by row. Median absolute error {median:.2f} mm against the "
        f"sonographer's measurement over {totals['n']} held-out images, mean "
        f"{totals['mae_mm']:.2f} mm, worst {totals['worst_mm']:.2f} mm - and "
        f"{totals['model_mae_mm']:.2f} mm against the truth mask's own circumference, which "
        "is this model's contribution rather than the dataset's drawing convention. **Do "
        f"not choose these weights on the Dice**: it is {totals['dice']:.4f}, it orders the "
        f"cases differently from the millimetres, and {wrong['n_over']} of the "
        f"{wrong['n_publishable']} cases scoring {wrong['dice_at_least']:.2f} or better are "
        f"still wrong by more than {wrong['matters_mm']:.0f} mm. **CC BY 4.0, not MIT**, "
        "because the data is."
    )
    wrapped = textwrap.wrap(description, width=84)
    body = [f'            "{line} "' for line in wrapped[:-1]]
    body.append(f'            "{wrapped[-1]}"')
    return "\n".join(
        [
            '    "hc18-unet": Published(',
            '        repo="maju116/platypus-weights",',
            '        filename="hc18-unet.safetensors",',
            '        revision="<the commit git ls-remote reports after the upload>",',
            "        description=(",
            *body,
            "        ),",
            "    ),",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="examples/data/hc18")
    parser.add_argument("--out", default="weights")
    parser.add_argument("--prepared", default="measurements/hc18")
    # Every argument below shapes the model, and every default is the example's own.
    # `tests/test_examples.py` holds the two sets equal.
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
        help="how many truth masks the no-model control reads for the card's dataset figures",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=4.0,
        help="refuse to write if the MEDIAN absolute error exceeds this many millimetres",
    )
    arguments = parser.parse_args()

    script = example_module()
    root = Path(arguments.data)
    prepared = Path(arguments.prepared)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    print("--- filling the drawn outlines, which is what anyone holding HC18 must do first ---")
    cases, table, open_curves = script.prepare(root, prepared)
    print(f"  {len(table)} usable cases, {len(open_curves)} excluded for an unclosed curve")

    # Before training, because it needs no model and because a card whose dataset figures
    # cannot be measured should fail before the GPU is spent rather than after. This is the
    # number that separates the dataset's drawing convention from the model's own error, and
    # it is measured here rather than typed so that the card has no figure a reader cannot
    # trace to this run.
    print("\n--- the measurement pipeline against the sonographer's own ellipse ---")
    floor = script.pipeline_control(
        cases, table, arguments.controls, root / "training_set" / "training_set"
    )
    print(
        f"  n={floor['n']}  bias {floor['bias_mm']:+.3f} mm from a line "
        f"{floor['line_width_px']:.3f} px wide; pi*w*pixel predicts "
        f"{floor['predicted_bias_mm']:+.3f}, correlation {floor['correlation']:.4f}, "
        f"residual {floor['residual_mae_mm']:.3f} mm"
    )

    configuration = with_checkpoint(script.specification(cases, arguments), out)
    spec = pp.from_dict(configuration)
    engine = pp.build_engine(spec, num_workers=arguments.workers)

    started = time.time()
    history = engine.fit(verbose=True)["skull"]
    minutes = (time.time() - started) / 60
    epochs = len(history.records)
    print(f"\ntrained in {minutes:.1f} min, {epochs} epochs")

    # The engine's own division rather than this script's arithmetic over the CSV: the
    # fractions and the `group_by` are in the configuration, but what the split *came out*
    # as depends on how the examinations happened to fall, and a card stating the fractions
    # instead of the counts would be stating an intention. `dataset` wants the model object,
    # which is the specification's rather than a name.
    counts = {
        name: len(engine.dataset(spec.models[0], name)) for name in ("train", "validation", "test")
    }
    print(f"  split: {counts}")

    print("\n--- per case, in both units, on the grid the files are on ---")
    rows = script.measure(engine, cases, table, "test")
    totals = script.summary(rows)
    gap = script.disagreement(rows)
    quantiles = spread(rows)
    wrong = published_but_wrong(rows)
    print(
        f"  n={totals['n']}  dice {totals['dice']:.4f}  median {quantiles['median_mm']:.3f} mm  "
        f"mae {totals['mae_mm']:.3f} mm  p90 {quantiles['p90_mm']:.3f}  "
        f"worst {totals['worst_mm']:.3f} mm"
    )
    print(
        f"  spearman {gap['spearman']:.3f}; of {wrong['n_publishable']} cases at dice "
        f">= {PUBLISHABLE_DICE}, {wrong['n_over']} are over {MATTERS_MM:.0f} mm"
    )

    # The control for this script's own reading of the prediction: the engine scores the same
    # split on the same grid by its own code, so a Dice computed here that disagrees with it
    # means the channels or the cases were read wrongly - and every millimetre above would be
    # measured on the wrong mask. Seven decimals rather than two, because the two decimals a
    # card prints would hide exactly the discrepancy worth catching.
    engine_rows = engine.evaluate_cases("skull", split="test", at="source")
    by_case = {row["case"]: row for row in engine_rows}
    worst_gap = max(abs(row["dice"] - by_case[row["case"]]["dice"]) for row in rows)
    print(f"\n  own Dice against the engine's, over {len(rows)} cases: {worst_gap:.2e}")
    if worst_gap > 1e-5:
        raise SystemExit(
            f"this script's Dice and the engine's disagree by {worst_gap:.3e} on at least one "
            "case, so the predictions are not being read the way the engine reads them and "
            "the millimetres above are measured on the wrong masks. Nothing was written."
        )

    measured = {
        "summary": totals,
        "spread": quantiles,
        "pipeline_control": floor,
        "disagreement": gap,
        "published_but_wrong": wrong,
        "split": counts,
        "open_curves": open_curves,
        "epochs": epochs,
        "minutes": minutes,
        "trained_with": pp.__version__,
        "cases": rows,
    }
    (out / "hc18-unet-measured.json").write_text(
        json.dumps(measured, indent=2, default=float) + "\n"
    )
    print(f"\nwrote {out / 'hc18-unet-measured.json'}")

    # The refusal `publish_dsbowl_weights.py` taught: weights whose measurement did not come
    # out are not published, because the card *is* the measurement. A run landing here below
    # the floor has still written its numbers above, so the failure is diagnosable rather
    # than silent. The floor is in millimetres because that is what these weights are for -
    # refusing on a Dice would be the mistake this whole card is about.
    # On the MEDIAN, and calibrated against runs that happened rather than against a number
    # somebody remembered. The first version of this guard floored the *mean* at 4.0 mm and
    # cited "2.57 mm" for the previous run; the previous run's mean was 5.66 mm, so the
    # figure was invented and the guard refused a run that was slightly better than the one
    # it was written from. §3C, inside a refusal - where a wrong number does the most damage,
    # because it tells the next reader what a healthy run looks like.
    #
    # Measured, two runs of this configuration: medians 2.00 and 2.19 mm, means 5.66 and
    # 5.22. The median is what the guard reads, because the mean describes a tail that is a
    # property of the data rather than of the run, and the 1-epoch smoke test - a genuinely
    # broken run - sits at 60 mm, four orders of magnitude of headroom away from either.
    if quantiles["median_mm"] > arguments.floor:
        raise SystemExit(
            f"median absolute error {quantiles['median_mm']:.3f} mm is above the floor of "
            f"{arguments.floor:.1f} mm, so nothing was written. Two runs of this "
            "configuration measured 2.00 and 2.19 mm; a number far above that is a broken "
            "run rather than a worse model, and publishing it would publish the breakage. "
            f"For reference this run's mean was {totals['mae_mm']:.3f} mm, which is "
            "expected to be two to three times the median here - the distribution has a "
            "long tail and that is the finding rather than a fault."
        )

    # Built before the weights are written, and that order is the point: the first version
    # of this script exported the file and then crashed formatting the card, which leaves a
    # `.safetensors` on disk with nothing describing it - the exact state `weights.py` exists
    # to prevent. A card that cannot be built now means nothing was written at all.
    text = card(measured) + "\n"

    path = engine.export_weights(
        "skull",
        out / "hc18-unet",
        data="HC18, a 60/20/20 split of the training set grouped by examination",
        data_source=SOURCE,
        data_licence="CC BY 4.0",
        citation=CITATION,
        weights_licence="CC BY 4.0",
        trained_with=f"pyplatypus {pp.__version__}",
        split={**counts, "group_by": script.GROUP_BY, "seed": arguments.seed},
        validation=[{"metric": "dice", "mean": totals["dice"], "worst": totals["dice_worst"]}],
        millimetres={
            "mae_mm": totals["mae_mm"],
            "bias_mm": totals["bias_mm"],
            "worst_mm": totals["worst_mm"],
            "model_mae_mm": totals["model_mae_mm"],
            "model_bias_mm": totals["model_bias_mm"],
        },
        disagreement={
            "spearman": gap["spearman"],
            "publishable_dice": PUBLISHABLE_DICE,
            "n_publishable": wrong["n_publishable"],
            "n_over_2mm": wrong["n_over"],
        },
        excluded=sorted(open_curves),
    )
    (out / "hc18-unet-card.md").write_text(text)

    print(f"\nwrote {path}")
    print(f"      {path.with_suffix('.json')}")
    print(f"      {out / 'hc18-unet-card.md'}")
    print(
        "\nNothing was uploaded. The upload is the author's: see weights/UPLOAD.md. Then the "
        "registry entry, pinned to the Hub commit that upload produces:"
    )
    print()
    print(registry_entry(measured, (arguments.height, arguments.width)))


if __name__ == "__main__":
    main()
