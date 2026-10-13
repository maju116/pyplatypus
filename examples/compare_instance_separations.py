"""Both separation signals and five decodings, from one configuration file.

    python examples/compare_instance_separations.py --epochs 60

**This script is the test of whether `task: instance_segmentation` earned its place**, and
it is deliberately the shortest file in this directory: it says where the data is - by
naming a YAML file that says it - and the engine does the rest. The two scripts beside it,
`train_instance_separation.py` and `train_instance_flows.py`, each assemble a model, a
dataset, a loss and a decoding by hand, because until `pyplatypus#160` the head's width
could not be derived from a specification. They are kept as the record of the measurements
they produced; this is the path to copy.

What one run prints, from `examples/data_science_bowl_instances.yaml`: two models - one
trained on an interior-and-rim target, one on a flow field - and five rows, because `decode`
is a list and a second decoding of 134 images costs 1.41 s against a second training run.
**A whole run of this file is 185 minutes on this machine** - 98.8 for the rim model's 120
epochs and 86.5 for the flow model's 60 - so a decoding is a three-thousandth of a run. The
figure this paragraph used to quote, 16 minutes, was measured on a much shorter run and was
restated here as the cost of *these* two; the ratio was the point and the ratio got larger.

**Five rows and not six.** `pyplatypus#215` asked for every separation against every decode,
and when it was written that meant three signals. `hv` is not one of them: its markers lose
to a one-pixel erosion on perfect inputs, 0.9382 against 0.9807, so `pyplatypus#213` closed
measured-and-no and `hv` is deliberately not a name in the specification. A sixth row here
would promise a mechanism that was measured as absent.

**Read the `connected_components` rows first.** That decoding ignores the separation signal
entirely, so it is what the signal's own row has to beat; and it is capped, by a measurement
rather than by an argument, at 0.7655 - a *perfect* semantic mask scores that, because
touching nuclei are one region. A row between the published weights' 0.4888 and that number
has improved the mask and said nothing about separation. `connected_components` is therefore
never dropped by `--decode`, because a table without it cannot be read.

**And the answer this configuration gave is not flattering, which is the point.** Run on
2026-10-13: the rim route is worth **+0.0676** over its own connected-components row and the
flow route **+0.0661** over its own, while substituting a perfect mask is worth +0.2398. So
the mask is the limit by a factor of 3.6 and the choice between the two signals is worth
almost nothing - both numbers reproduce measurements taken separately, +0.0640 in `#159`'s
sweep and +0.0680 in `#212`. This file exists to make that reproducible from one command
rather than to promise otherwise.

**The noise floor comes out of this file rather than from somewhere else.** The flow model
was configured identically in two runs of it and its average precision moved 0.0112 between
them, where `#159`'s sweep measured 0.0074 - so read any gap smaller than 0.0112 as nothing,
including most of what separates the two routes above. `pyplatypus#159`, `#212`, `#213`,
`#215`; `JOURNAL.md` §4bo and §4bq.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyplatypus as pp

HERE = Path(__file__).resolve().parent
CONFIGURATION = HERE / "data_science_bowl_instances.yaml"

#: The columns, in the order a reader needs them: what was trained, how it was decoded, the
#: mask's own number, then the instance score with the counts that explain it. The object
#: counts are not decoration - an average precision cannot tell a route that mislabels from
#: one that shatters, and the count is what does.
#:
#: **`class dice` is read down a model's own rows and never across two models**, and it was
#: called `fg dice` here until 2026-10-13, which is a stronger claim than the number can
#: carry. It is the engine's `dice` metric: a mean over the class channels the model's head
#: actually has, which is three for `rim` - background, interior and the rim band - and two
#: for `flows`, whose metrics are shown the class block alone. So the rim model's number is
#: averaged against a thin band that scores 0.6857 where its background and interior score
#: 0.9852 and 0.8721, and reading it beside a flow model's is reading two different
#: quantities. Measured by identical code on both, the binary foreground Dice is 0.9159 for
#: the rim model against 0.9097 for the flow model - 0.0062 apart, where this column shows
#: them 0.09 apart and in the other direction.
COLUMNS = [
    ("model", "model", "{}"),
    ("separation", "separation", "{}"),
    ("decode", "decode", "{}"),
    ("class dice", "dice", "{:.4f}"),
    ("AP", "average_precision", "{:.4f}"),
    ("objects", "predicted_objects", "{}"),
    ("truth", "truth_objects", "{}"),
]

#: The yardstick, which `--decode` is not allowed to remove. See the module docstring.
YARDSTICK = "connected_components"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGURATION), help="the configuration to run")
    parser.add_argument(
        "--epochs", type=int, default=None, help="override every model's epoch count"
    )
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or leave it to torch")
    parser.add_argument("--split", default="validation", help="which split to score the table on")
    parser.add_argument(
        "--separation",
        action="append",
        choices=sorted(row["name"] for row in pp.available_separations()),
        help="run only the models trained on this signal; repeatable. A SELECTOR rather "
        "than a definition - see `selected`",
    )
    parser.add_argument(
        "--decode",
        action="append",
        choices=sorted(row["name"] for row in pp.available_decodings()),
        help=f"score only these decodings; repeatable. `{YARDSTICK}` stays in whatever this "
        "says, because it is the row the others are read against",
    )
    parser.add_argument(
        "--expect-cases",
        default=None,
        help="a JSON file of the run this is compared against; its case names must be the "
        "scored split exactly, which is a stronger control than any score",
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="a directory for the figure: the best and the worst row on one case, as found, "
        "missed and invented objects",
    )
    return parser.parse_args()


def selected(config: dict, arguments: argparse.Namespace) -> dict:
    """The configuration, narrowed by the command line but never defined by it.

    `--epochs` changes what the file says. `--separation` and `--decode` do not: they
    **select a subset of the rows the file already describes**, so the published table is
    what one bare invocation prints and a flag can only ever show less of it. Everything
    that makes this comparison a comparison - the split, the architecture, the input size,
    the loss - stays in the file, where it can be read beside the numbers it produced.

    `pyplatypus#215` asked for "`--separation` and `--decode` driving it", which it wrote
    before `decode` became a list in the specification. One command already produces every
    row, so the flags are what a person debugging one row wants, and giving them the power
    to invent a row would make the table depend on a command nobody wrote down.
    """
    models = config["models"]

    if arguments.separation:
        wanted = set(arguments.separation)
        models = [model for model in models if model["separation"]["name"] in wanted]
        if not models:
            raise SystemExit(
                f"--separation {sorted(wanted)} selects none of the models in "
                f"{arguments.config}, which are "
                f"{sorted(m['separation']['name'] for m in config['models'])}"
            )

    if arguments.decode:
        wanted = set(arguments.decode) | {YARDSTICK}
        narrowed = []
        for model in models:
            blocks = [block for block in model["decode"] if block["name"] in wanted]
            if not blocks:  # pragma: no cover - YARDSTICK is in every model's list
                continue
            narrowed.append({**model, "decode": blocks})
        models = narrowed

    if arguments.epochs is not None:
        models = [{**model, "epochs": arguments.epochs} for model in models]

    return {**config, "models": models}


def check_cases(engine: pp.InstanceEngine, spec, split: str, expected_file: str) -> None:
    """The control `pyplatypus#215` asks this script to inherit rather than reinvent.

    The split has to be the split the comparison names, **by case name**. Two different sets
    of 134 images give numbers that look comparable and are not, and that has already cost a
    near-published figure here: a model card's "134 held-out images" and the 134 case
    directories on this disk are a different 134, and the discrepancy was 0.01 of Dice -
    exactly what a rounding difference would also look like.
    """
    expected = {row["case"] for row in json.loads(Path(expected_file).read_text())["cases"]}
    samples = engine.dataset(spec.models[0], split).samples
    got = {Path(sample.key).name for sample in samples}
    if got != expected:
        raise SystemExit(
            f"the {split} split is {len(got)} cases and {len(got - expected)} of them are not "
            f"in the run this is compared against ({len(expected)} cases). Two different sets "
            "of images give numbers that look comparable and are not - do not read the score"
        )
    print(f"control: the {split} split is exactly the {len(expected)} cases of {expected_file}")


def draw(engine: pp.InstanceEngine, spec, split: str, rows: list[dict], into: Path) -> Path:
    """The best row and the worst row, on **one** case, as found, missed and invented objects.

    One case rather than two, because two pictures of two images would show the spread
    between images - 0.054 of Dice on this data - and read as the spread between methods.
    The case drawn is the one where the two rows disagree most **among the half of the split
    with the most objects**, which is what a figure can say and a mean cannot; its name, its
    object count and both scores are printed, so the picture is tied to numbers rather than
    chosen for looking good.

    Drawn at an IoU of 0.5 while the table's average precision is the mean over ten
    thresholds from 0.5 to 0.95. So this is the most forgiving threshold there is: an object
    the picture already calls missed is missed at every threshold the number averages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyplatypus.style import AGREEMENT_COLOURS

    ranked = sorted(rows, key=lambda row: row["average_precision"])
    worst, best = ranked[0], ranked[-1]

    per_case = {}
    objects = {}
    for row in (best, worst):
        key = (row["model"], row["decode"])
        scored = engine.evaluate_cases(row["model"], split, decode=row["decode"])
        per_case[key] = {case["case"]: case["average_precision"] for case in scored}
        objects.update({case["case"]: case["truth"] for case in scored})

    shared = set(per_case[(best["model"], best["decode"])]) & set(
        per_case[(worst["model"], worst["decode"])]
    )
    spread = {
        case: per_case[(best["model"], best["decode"])][case]
        - per_case[(worst["model"], worst["decode"])][case]
        for case in shared
    }
    # The largest disagreement, but only among the cases with **enough objects to disagree
    # about**. Measured 2026-10-13: ranking by the raw gap picked an image holding three
    # nuclei, where one object is a third of the score, and the figure showed almost
    # nothing. An average precision over few objects is high variance, so "where the methods
    # differ most" and "where the picture shows how they differ" are different questions -
    # §4's entry about a probe answering the one next to the one it is named after.
    crowded = sorted(shared, key=lambda name: objects[name])[len(shared) // 2 :]
    case = max(crowded, key=lambda name: abs(spread[name]))

    panels = []
    for row in (best, worst):
        model = next(m for m in spec.models if m.name == row["model"])
        dataset = engine.dataset(model, split)
        # `evaluate_cases` walks the split in the dataset's own order and never shuffles, so
        # a case's position in that list is the index to draw.
        order = [
            c["case"] for c in engine.evaluate_cases(row["model"], split, decode=row["decode"])
        ]
        index = order.index(case)
        prediction = engine.predict_instances(row["model"], split, decode=row["decode"])[index]
        panels.append(
            (
                row,
                pp.overlay_instance_agreement(
                    dataset[index][0], prediction, dataset.labels_of(index)
                ),
                per_case[(row["model"], row["decode"])][case],
            )
        )

    into.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for axis, (row, picture, score) in zip(axes, panels):
        axis.imshow(picture)
        axis.set_title(
            f"{row['separation']} + {row['decode']}\nthis case AP {score:.4f}  "
            f"(split {row['average_precision']:.4f})",
            fontsize=10,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=colour, label=label.replace("_", " "))
        for label, colour in AGREEMENT_COLOURS.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    # The case name is a 64-character hash, which as a title is a wall rather than a label.
    # Sixteen characters is what `train_instance_separation.py` prints and enough to find the
    # directory; the whole name goes to stdout below, where it can be copied.
    figure.suptitle(f"{case[:16]}  -  objects matched at IoU 0.5, the way the metric matches them")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = into / "instance-separations-best-worst.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(
        f"\nfigure: {path}\n  case {case} ({objects[case]} objects), where the two rows "
        f"differ by {abs(spread[case]):.4f} of average precision - the largest gap among "
        f"the {len(crowded)} most crowded of the {len(shared)} cases scored by both"
    )
    return path


def print_table(rows: list[dict]) -> None:
    widths = [
        max(len(title), *(len(fmt.format(row[key])) for row in rows)) for title, key, fmt in COLUMNS
    ]
    header = "  ".join(title.ljust(width) for (title, _, _), width in zip(COLUMNS, widths))
    print(f"\n{header}")
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                fmt.format(row[key]).ljust(width) for (_, key, fmt), width in zip(COLUMNS, widths)
            )
        )


def main() -> None:
    import yaml

    arguments = parse()
    config = selected(yaml.safe_load(Path(arguments.config).read_text()), arguments)
    spec = pp.from_dict(config, source=arguments.config)

    engine = pp.build_engine(spec, device=arguments.device)
    for model in spec.models:
        print(
            f"{model.name}: {model.signal.value}, head "
            f"{model.head_channels(spec.data.n_class)} channels, decoded "
            f"{len(model.decodings)} ways"
        )
    if arguments.expect_cases:
        check_cases(engine, spec, arguments.split, arguments.expect_cases)

    engine.fit(verbose=True)

    rows = engine.evaluate(arguments.split)
    print_table(rows)
    print(
        "\nThe connected_components rows ignore the separation signal, so they are what the "
        "other rows\nhave to beat - and a perfect semantic mask scores 0.7655 through that "
        "decoding on this data,\nwhich is the ceiling every row here is under."
    )
    print(
        "`class dice` is a mean over each model's OWN class channels - three for rim, two "
        "for flows -\nso it is read down a model's rows and never across two models; the "
        "two signals' masks are\ncloser than that column makes them look. See COLUMNS."
    )

    signal = {}
    for row in rows:
        signal.setdefault(row["model"], {})[row["decode"]] = row["average_precision"]
    for model, scored in signal.items():
        if YARDSTICK in scored and len(scored) > 1:
            own = max(score for name, score in scored.items() if name != YARDSTICK)
            print(
                f"  {model}: the signal is worth {own - scored[YARDSTICK]:+.4f} over "
                f"{YARDSTICK} on its own mask"
            )
    print(
        "  two runs of one configuration differ by 0.0074, so read any gap smaller than that\n"
        "  as nothing - §4 has an entry about an invented denominator"
    )

    best = engine.best_model("average_precision", arguments.split)
    print(f"Best by instance average precision: {best}")

    if arguments.figures:
        draw(engine, spec, arguments.split, rows, Path(arguments.figures))


if __name__ == "__main__":
    main()
