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

#: Every script in `examples/`, read from the directory rather than named here. The list this
#: replaces was six names somebody had to remember to extend, which is the same oversight in
#: smaller form as the one this file was written for: nothing was watching the examples at all.
SCRIPTS = sorted(path.stem for path in EXAMPLES.glob("*.py"))


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
    assert "cldice" in named, (
        "the point of this example is a whole-mask metric on a tiled run; `evaluate_cases` "
        "reassembles the tiles before measuring it"
    )
    assert config["models"][0]["splits"], "the example exists to exercise tiling"


def test_detection_configuration_is_accepted():
    import detect_blood_cells as script

    arguments = _declared_arguments("detect_blood_cells")
    splits = {name: Path(f"{name}.csv") for name in ("train", "validation", "test")}
    config = script.configuration(splits, arguments)
    _validate(config)
    assert config["task"] == "object_detection"


def test_the_instance_baseline_asks_for_the_published_weights_and_no_training():
    """The script that measures the instance baseline must not train anything.

    It exists to score weights that are already published, and a configuration that trained
    would quietly produce a different model and a baseline nobody could reproduce. `fit:
    false` and the registry name are the two things that make it the measurement it claims to
    be, so both are asserted rather than read.
    """
    import score_dsbowl_instances as script

    config = script.specification("train.csv", "validation.csv", 256)
    _validate(config)

    model = config["models"][0]
    assert model["fit"] is False, "this script measures, it does not train"
    assert model["weights"] == script.WEIGHTS
    assert model["input_shape"] == [256, 256], (
        "the published weights were trained at 256; the sidecar refuses another shape, and "
        "this test fails here rather than in the loader"
    )


def test_published_fives_weights_configuration_is_accepted():
    """The published weights' configuration is the example's, and the flags have to match.

    `publish_fives_weights.py` hands its own `arguments` to the example's `specification()`,
    so every flag that function reads has to be declared here too - a coupling nothing else
    would notice until six hours into a run. `_declared_arguments` reads the publishing
    script's parser, so a flag the example grows and the publisher does not raises
    `AttributeError` in this test rather than in the dark.
    """
    import publish_fives_weights as script
    import segment_retinal_vessels as example

    arguments = _declared_arguments("publish_fives_weights")
    splits = {name: Path(f"{name}.csv") for name in ("train", "test")}
    config = script.with_checkpoint(example.specification(splits, arguments), Path("weights"))
    _validate(config)

    watched = {entry["name"]: entry.get("monitor") for entry in config["models"][0]["callbacks"]}
    assert watched["model_checkpoint"] == watched["early_stopping"], (
        "the checkpoint and the stopping rule have to watch the same metric: `restore_best` "
        "hands back the epoch the stopping rule chose, so two monitors means the file on disk "
        "and the weights that get published come from different epochs"
    )


def test_the_card_names_the_weakest_group_in_each_table():
    """The card's one sentence of argument reads two tables, and swapping them would invert it.

    Everything else on the card is a number substituted into a table. This sentence is the
    conclusion - which group is weakest over all images, and which over the ones graded clean -
    and it is the part a reader quotes. The two names are deliberately different here, so
    reading either table twice fails rather than coincidentally agreeing.
    """
    import publish_fives_weights as script

    distribution = {"n": 50, "mean": 0.5, "sd": 0.1, "min": 0.1}
    measured = {
        "seed": 1,
        "epochs": 40,
        "epochs_run": 24,
        "per_case": {
            "dice": {**distribution, "mean": 0.8},
            "cldice": {**distribution, "mean": 0.7},
        },
        "by_disease": {
            "dice": {
                "alpha": {**distribution, "mean": 0.50},
                "beta": {**distribution, "mean": 0.90},
            }
        },
        "by_disease_clean": {
            "dice": {
                "alpha": {**distribution, "mean": 0.95},
                "beta": {**distribution, "mean": 0.60},
            }
        },
        "by_quality": {
            "dice": {
                "graded_clean": {**distribution, "n": 133},
                "graded_poor": {**distribution, "n": 67},
            }
        },
    }

    paragraph = " ".join(script.card(measured).split())
    assert (
        "`alpha` is the weakest group over all images and `beta` over the clean ones" in paragraph
    ), (
        "the sentence has to read the all-images table and the graded-clean table, in that "
        "order; this is the claim the card is quoted for"
    )


def _graded(flagged: list[str]) -> dict[str, dict[str, int]]:
    """200 test images keyed as FIVES names them, with `flagged` the low-contrast ones."""
    keys = [f"{number}_{disease}" for disease in "ADGN" for number in range(1, 51)]
    return {
        key: {"illumination_contrast": 1, "blur": 1, "low_contrast": int(key not in flagged)}
        for key in keys
    }


def test_the_quality_polarity_check_can_refuse():
    """The card's whole argument rests on 1 meaning *good*, so the check has to reject the
    other reading rather than establish the polarity it assumed."""
    import publish_fives_weights as script

    # What the sheet holds: 15 images flagged for low contrast, 14 of them glaucoma.
    flagged = [f"{number}_G" for number in range(1, 15)] + ["1_N"]
    script.check_polarity(_graded(flagged))

    inverted = {
        key: {**grades, "low_contrast": 1 - grades["low_contrast"]}
        for key, grades in _graded(flagged).items()
    }
    with pytest.raises(SystemExit, match="do not publish"):
        script.check_polarity(inverted)


def test_the_examples_were_found():
    """An empty glob parametrizes nothing, and zero tests read exactly like six passing ones."""
    assert len(SCRIPTS) >= 8, SCRIPTS


def test_the_separation_ceiling_runs_and_beats_connected_components(tmp_path, capsys):
    """Run it, rather than inspect it.

    The two numbers this script exists to put side by side are the only thing it produces, and
    neither is reachable by reading its `parser.add_argument` calls. So the test builds two
    cases whose objects touch - the case the whole milestone is about - and checks that the
    ceiling is above what connected components manages on the same pictures. On data this
    clean the separated answer is exact and the naive one merges every pair, so the gap is
    large on purpose: a test that only asserted "different" would pass for a script that had
    them the wrong way round.
    """
    import numpy as np
    from PIL import Image

    script = __import__("separation_ceiling")
    for case in ("one", "two"):
        masks = tmp_path / case / "masks"
        masks.mkdir(parents=True)
        for index, columns in enumerate(((1, 6), (6, 11))):
            mask = np.zeros((9, 11), np.uint8)
            mask[2:7, columns[0] : columns[1]] = 255
            Image.fromarray(mask).save(masks / f"{index}.png")

    sys.argv = ["separation_ceiling.py", "--data", str(tmp_path)]
    script.main()
    printed = capsys.readouterr().out

    def row(label: str) -> float:
        """The first number on the line with this label.

        Not a column index: the labels have different word counts, so position would make
        this test depend on the width of a name rather than on the number beside it.
        """
        import re

        line = next(l for l in printed.splitlines() if l.strip().startswith(label))
        return float(re.search(r"\d+\.\d+", line).group())

    assert row("rim + nearest_seed") == pytest.approx(1.0), (
        "the parts come apart exactly on data this clean"
    )
    assert row("connected_components") < 0.2, "and connected components merges both pairs"
    # The watershed is the second decoding of the same parts. On a ceiling run its surface is
    # flat, so it has nothing to climb - it must not come apart *worse* than the baseline it
    # is meant to improve on, which is the only thing this run can say about it.
    assert row("rim + watershed") > row("connected_components")
    assert "erased by the erosion, rescued by fallback  0" in printed
    assert "floor and not as a comparison" in printed, (
        "the watershed row without its caveat is a misleading number in the most-read place"
    )


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_example_imports(name):
    """Importing is its own assertion: a script naming a function the package has dropped
    fails here rather than in front of whoever copied it."""
    __import__(name)
