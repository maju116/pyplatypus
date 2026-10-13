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
is a list and a second decoding of 134 images costs 1.41 s against 16 minutes for a second
training run.

**Read the `connected_components` rows first.** That decoding ignores the separation signal
entirely, so it is what the signal's own row has to beat; and it is capped, by a measurement
rather than by an argument, at 0.7655 - a *perfect* semantic mask scores that, because
touching nuclei are one region. A row between the published weights' 0.4888 and that number
has improved the mask and said nothing about separation.

**And the answer this configuration gives has already been measured.** The rim route is
worth +0.0640 over its own connected-components row and the flow route +0.0680 over its
own, a difference of 0.0040 against the 0.0074 two runs of one configuration differ by -
while substituting a perfect mask is worth +0.2398. So the mask is the limit by a factor of
3.6, and this file exists to make that reproducible from one command rather than to promise
otherwise. `pyplatypus#159`, `#212`, `#213`; `JOURNAL.md` §4bo and §4bq.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyplatypus as pp

HERE = Path(__file__).resolve().parent
CONFIGURATION = HERE / "data_science_bowl_instances.yaml"

#: The columns, in the order a reader needs them: what was trained, how it was decoded, the
#: mask's own number, then the instance score with the counts that explain it. The object
#: counts are not decoration - an average precision cannot tell a route that mislabels from
#: one that shatters, and the count is what does.
COLUMNS = [
    ("model", "model", "{}"),
    ("separation", "separation", "{}"),
    ("decode", "decode", "{}"),
    ("fg dice", "dice", "{:.4f}"),
    ("AP", "average_precision", "{:.4f}"),
    ("objects", "predicted_objects", "{}"),
    ("truth", "truth_objects", "{}"),
]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGURATION), help="the configuration to run")
    parser.add_argument(
        "--epochs", type=int, default=None, help="override every model's epoch count"
    )
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or leave it to torch")
    parser.add_argument("--split", default="validation", help="which split to score the table on")
    return parser.parse_args()


def overridden(config: dict, arguments: argparse.Namespace) -> dict:
    """The configuration with the one thing a command line should be able to change.

    `--epochs` only. Everything else that makes this comparison a comparison - the split,
    the architecture, the input size, the loss - belongs in the file, where it can be read
    beside the numbers it produced; a flag for each of them would make the published table
    depend on a command nobody wrote down.
    """
    if arguments.epochs is None:
        return config
    return {
        **config,
        "models": [{**model, "epochs": arguments.epochs} for model in config["models"]],
    }


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
    config = overridden(yaml.safe_load(Path(arguments.config).read_text()), arguments)
    spec = pp.from_dict(config, source=arguments.config)

    engine = pp.build_engine(spec, device=arguments.device)
    for model in spec.models:
        print(
            f"{model.name}: {model.signal.value}, head "
            f"{model.head_channels(spec.data.n_class)} channels, decoded "
            f"{len(model.decodings)} ways"
        )
    engine.fit(verbose=True)

    rows = engine.evaluate(arguments.split)
    print_table(rows)
    print(
        "\nThe connected_components rows ignore the separation signal, so they are what the "
        "other rows\nhave to beat - and a perfect semantic mask scores 0.7655 through that "
        "decoding on this data,\nwhich is the ceiling every row here is under."
    )
    best = engine.best_model("average_precision", arguments.split)
    print(f"Best by instance average precision: {best}")


if __name__ == "__main__":
    main()
