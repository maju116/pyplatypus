"""The example scripts' configurations, validated without training anything.

These four scripts are the code a reader copies, and nothing ran them. `n_class` was removed
from the specification on one day and `task` became required on another, and all four kept
carrying the first while three lacked the second - broken for three days behind a green suite,
because ruff reads their layout and the test suite had never heard of them.

So: build each script's configuration and hand it to the loader. `check_paths=False` because
the point is the shape of the request, not whether this machine holds BCCD. Nothing is trained
and nothing is downloaded.

The arguments come from each script's own `parser.add_argument` calls rather than from a list
written here, so a flag that is renamed is followed and a flag the configuration reads but the
parser never declares raises `AttributeError` in this file. That second case has happened
before: `--n-class` was once parsed and never used, and the reverse is the same mistake.
"""

from __future__ import annotations

import ast
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from pyplatypus import from_dict

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

PLACEHOLDER = {"int": 1, "float": 0.5, "str": "x", None: "x"}


def _declared_arguments(name: str) -> Namespace:
    """Every `--flag` the script declares, with its default, as a Namespace.

    A flag with no default is required, and the script would not run without it, so it gets a
    placeholder of the declared type. Nothing here knows the names of the flags.
    """
    tree = ast.parse((EXAMPLES / f"{name}.py").read_text())
    values: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if not isinstance(flag, ast.Constant) or not str(flag.value).startswith("--"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        dest = str(flag.value)[2:].replace("-", "_")
        if "default" in keywords:
            values[dest] = ast.literal_eval(keywords["default"])
            continue
        kind = keywords.get("type")
        kind = kind.id if isinstance(kind, ast.Name) else None
        values[dest] = [] if "nargs" in keywords else PLACEHOLDER[kind]
    assert values, f"{name} declares no flags; the parser is not where this expects it"
    return Namespace(**values)


def _validate(config: dict) -> None:
    from_dict(config, check_paths=False)


def test_architecture_comparison_configuration_is_accepted():
    import compare_dsbowl_architectures as script

    arguments = _declared_arguments("compare_dsbowl_architectures")
    config = script.specification(
        Path("train.csv"), Path("validation.csv"), list(script.ARCHITECTURES), arguments
    )
    _validate(config)
    assert [entry["name"] for entry in config["models"]] == list(script.ARCHITECTURES)


def test_pretrained_comparison_configuration_is_accepted():
    import compare_pretrained_encoders as script

    arguments = _declared_arguments("compare_pretrained_encoders")
    # Every arm the script compares, including the ones that add an encoder: a configuration
    # the comparison would train but the loader would refuse is the failure this file exists
    # for, and the arms are where the unusual fields live.
    for _, extra in script.configurations("resnet34", 1e-4, 5):
        _validate(script.specification(arguments, 1, extra))


def test_published_weights_configuration_is_accepted():
    import publish_dsbowl_weights as script

    _validate(script.specification("train.csv", "validation.csv", 60, 256))


def test_retinal_vessel_configuration_is_accepted():
    import segment_retinal_vessels as script

    arguments = _declared_arguments("segment_retinal_vessels")
    splits = {name: Path(f"{name}.csv") for name in ("train", "test")}
    config = script.specification(splits, arguments)
    _validate(config)

    # The one the specification refuses, asserted here because this example's whole shape -
    # two scoring passes rather than one - follows from it.
    named = [metric["name"] for metric in config["models"][0]["metrics"]]
    assert "cldice" not in named, "a tiled run cannot measure a whole-mask metric"
    assert config["models"][0]["splits"], "the example exists to exercise tiling"


def test_detection_configuration_is_accepted():
    import detect_blood_cells as script

    arguments = _declared_arguments("detect_blood_cells")
    splits = {name: Path(f"{name}.csv") for name in ("train", "validation", "test")}
    config = script.configuration(splits, arguments)
    _validate(config)
    assert config["task"] == "object_detection"


@pytest.mark.parametrize(
    "name",
    [
        "compare_dsbowl_architectures",
        "compare_pretrained_encoders",
        "publish_dsbowl_weights",
        "detect_blood_cells",
        "segment_retinal_vessels",
    ],
)
def test_every_example_imports(name):
    """Importing is its own assertion: a script naming a function the package has dropped
    fails here rather than in front of whoever copied it."""
    __import__(name)
