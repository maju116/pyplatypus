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
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from pyplatypus import from_dict
from tests.conftest import filled_ellipse

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


def test_the_instance_configuration_file_is_accepted_and_says_what_it_compares():
    """The YAML this example exists to run, validated rather than only parsed.

    Three claims, because the file is the example: it asks for the instance task, it holds
    both separation signals, and every model's `decode` list carries
    `connected_components`. The last of those is what makes the table readable - it is the
    row that ignores the separation signal, so it is what the signal's own row has to beat -
    and a file that dropped it would still run and produce numbers nothing could be read
    against.
    """
    import yaml

    config = yaml.safe_load((EXAMPLES / "data_science_bowl_instances.yaml").read_text())
    _validate(config)
    spec = from_dict(config, check_paths=False)
    assert spec.task.value == "instance_segmentation"
    assert {model.signal.value for model in spec.models} == {"rim", "flows"}
    for model in spec.models:
        decodings = [block.name for block in model.decodings]
        assert "connected_components" in decodings, model.name
        assert decodings[0] != "connected_components", (
            f"{model.name} reads its own signal's decoding first; the yardstick is what it "
            "is compared against"
        )
    # The widths the engine derives, which is the line this whole task turns on.
    assert [model.head_channels(spec.data.n_class) for model in spec.models] == [3, 4]


def test_the_instance_table_does_not_call_its_dice_a_foreground_dice():
    """The column is a mean over each model's own classes, and the two models differ.

    This guards a label rather than a computation, because the label is what was wrong: the
    column was called `fg dice` and read across the two models, which made the rim model's
    mask look a tenth of a Dice worse than the flow model's. It is neither a foreground
    score nor one quantity - `dice` reduces over the class channels the metric is shown,
    with the background among them, and `rim`'s head is three classes against `flows`'
    two.

    Two assertions, because either alone passes for the wrong reason: the channel counts
    are what makes the column incomparable, and the title is what stops a reader comparing
    it. Measured 2026-10-13 - the binary foreground Dice, by identical code on both, is
    0.9159 and 0.9097, where this column read 0.8565 and 0.9476.
    """
    import yaml

    sys.path.insert(0, str(EXAMPLES))
    import compare_instance_separations as script

    config = yaml.safe_load((EXAMPLES / "data_science_bowl_instances.yaml").read_text())
    spec = from_dict(config, check_paths=False)

    seen = {}
    for model in spec.models:
        # What the metric is handed: the head, minus any channel in front of the classes.
        # `_MetricOnClasses` is given exactly this window, and `rim` has no vectors so it
        # is the whole head.
        start = int(model.separation.semantic_from)
        seen[model.signal.value] = model.head_channels(spec.data.n_class) - start
        assert all(getattr(metric, "include_background", True) for metric in model.metrics), (
            f"{model.name} drops the background, which would change what this column is"
        )
    assert seen == {"rim": 3, "flows": 2}, (
        f"the two models' metrics see {seen} class channels; the column is a mean over "
        "them, so it is one quantity only while those two numbers are equal"
    )

    title = next(title for title, key, _ in script.COLUMNS if key == "dice")
    assert "fg" not in title and "foreground" not in title, (
        f"the column is called {title!r}; it is a mean over each model's own classes, "
        "background included, and calling it a foreground score invites the comparison "
        "across models that the counts above say cannot be made"
    )


def test_the_instance_example_selects_rows_and_never_invents_one():
    """`--separation` and `--decode` narrow what the file describes and cannot add to it.

    A flag per *setting* would make the published table depend on a command nobody wrote
    down, so `--epochs` is still the only thing that changes what the file says. The two
    selectors `pyplatypus#215` asks for are a different kind: they pick a subset of the rows
    already in the file, which is what somebody debugging one row wants and is safe to have
    because the bare invocation remains the whole table.

    The invariant worth pinning is the one a reader of the published numbers depends on:
    **no selection can produce a (separation, decode) pair the file did not list.** Pinned
    over every subset of both flags rather than over an example of each, because the way
    this would break is one combination, not the common case.
    """
    from argparse import Namespace
    from itertools import chain, combinations

    import compare_instance_separations as script

    def flags(**kwargs):
        base = {"separation": None, "decode": None, "epochs": None, "config": "f.yaml"}
        return Namespace(**{**base, **kwargs})

    config = {
        "task": "instance_segmentation",
        "models": [
            {
                "name": "rim",
                "epochs": 60,
                "separation": {"name": "rim"},
                "decode": [{"name": "nearest_seed"}, {"name": "connected_components"}],
            },
            {
                "name": "flows",
                "epochs": 60,
                "separation": {"name": "flows"},
                "decode": [{"name": "follow_flows"}, {"name": "connected_components"}],
            },
        ],
    }
    allowed = {
        (model["separation"]["name"], block["name"])
        for model in config["models"]
        for block in model["decode"]
    }

    assert script.selected(config, flags()) == config, "no flag, no change"

    changed = script.selected(config, flags(epochs=2))
    assert [model["epochs"] for model in changed["models"]] == [2, 2]
    assert config["models"][0]["epochs"] == 60, "the original is left alone"

    separations = ["rim", "flows"]
    decodings = ["nearest_seed", "follow_flows", "connected_components"]

    def subsets(xs):
        return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

    seen = set()
    for chosen_separations in subsets(separations):
        for chosen_decodings in subsets(decodings):
            arguments = flags(
                separation=list(chosen_separations) or None,
                decode=list(chosen_decodings) or None,
            )
            narrowed = script.selected(config, arguments)
            pairs = {
                (model["separation"]["name"], block["name"])
                for model in narrowed["models"]
                for block in model["decode"]
            }
            assert pairs <= allowed, (
                f"{chosen_separations}/{chosen_decodings} invented {pairs - allowed}"
            )
            if chosen_separations:
                assert {m["separation"]["name"] for m in narrowed["models"]} <= set(
                    chosen_separations
                )
            # The yardstick is never selected away, because a table without it cannot be read.
            for model in narrowed["models"]:
                assert script.YARDSTICK in [block["name"] for block in model["decode"]], (
                    f"{chosen_decodings} dropped the yardstick from {model['name']}"
                )
            seen |= pairs
    assert seen == allowed, "some subset of the flags never reached a row the file lists"

    # Selecting nothing is a refusal rather than an empty table: a run that scores no rows
    # prints a green nothing, which reads exactly like a comparison that found no difference.
    with pytest.raises(SystemExit) as caught:
        script.selected(config, flags(separation=["hv"]))
    assert "selects none of the models" in str(caught.value)


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
    assert len(SCRIPTS) >= 9, SCRIPTS


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


# ----------------------------------------------------------------- the flow training script
#
# `#212`'s second half ran outside `Engine`, because a flow model's head is `2 + n_class`
# wide and the engine took its width from `spec.data.n_class`. `#160` derives it now, and
# `examples/compare_instance_separations.py` is the configuration-driven path - but this
# script stays, because the measurement in its docstring came out of it, and its own
# functions are still the only place *its* target layout is decided. The engine's
# equivalents are tested in `tests/data/test_instance_dataset.py`, and the two targets are
# compared against each other nowhere: that would be a test of two implementations agreeing
# rather than of either being right.


def _flow_script():
    import train_instance_flows as script

    return script


def test_the_flow_target_is_two_vectors_then_the_two_semantic_channels():
    """The layout is the contract between the loss, the mask and the decode."""
    import numpy as np

    script = _flow_script()
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[2:7, 1:6] = 1
    labels[2:7, 6:11] = 2

    target = script.build_target(labels, "diffusion")
    assert target.shape == (4, 9, 11)
    assert target.dtype == np.float32
    foreground = labels > 0
    assert np.array_equal(target[3] > 0.5, foreground), "channel 3 is the foreground"
    assert np.array_equal(target[2] > 0.5, ~foreground), "channel 2 is the background"
    assert np.allclose(target[2] + target[3], 1.0), "and the two are a one-hot, not two masks"


def test_the_mask_the_loss_uses_is_the_semantic_channel_and_not_the_flow_s_own_zeros():
    """The reason `build_target` carries a foreground channel at all.

    A flow field is exactly zero at every object's sink, so its non-zero voxels are the
    foreground **minus precisely the seeds** — and `#211`'s oracle measured the seeds as what
    limits this family of methods. Channel 3 has no such hole, and the difference is exactly
    one voxel per object.
    """
    import numpy as np

    script = _flow_script()
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[2:7, 1:6] = 1
    labels[2:7, 6:11] = 2
    target = script.build_target(labels, "diffusion")

    moving = np.hypot(target[0], target[1]) > 0
    assert int((target[3] > 0.5).sum()) - int(moving.sum()) == 2, "two objects, two sinks"


def test_resizing_an_instance_map_leaves_no_hole_in_the_numbering():
    """An object can vanish in a nearest-neighbour resize, and a gap would read as a missing
    object the truth does not contain — the metric counts objects by their values."""
    import numpy as np

    script = _flow_script()
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[4:20, 4:20] = 1
    labels[30, 30] = 2  # one pixel, which an eight-fold shrink is free to drop
    labels[40:56, 40:56] = 3

    small = script.resize_labels(labels, 8)
    present = sorted(int(v) for v in np.unique(small) if v)
    assert present == list(range(1, len(present) + 1)), present
    assert len(present) < 3, "the premise: the one-pixel object does not survive this shrink"


def test_the_angle_metric_reads_zero_for_agreement_and_one_eighty_for_reversal():
    import numpy as np

    script = _flow_script()
    truth = np.zeros((2, 4, 4), dtype=np.float32)
    truth[0] = 1.0
    where = np.ones((4, 4), dtype=bool)

    assert script.angles_between(truth, truth, where).max() == pytest.approx(0.0, abs=1e-4)
    assert script.angles_between(-truth, truth, where).min() == pytest.approx(180.0, abs=1e-3)
    assert script.angles_between(truth, truth, np.zeros((4, 4), dtype=bool)).size == 0


def test_the_dataset_augments_the_label_map_and_builds_the_flow_target_afterwards(tmp_path):
    """`#210`'s rule, pinned in the script that depends on it.

    The augmenter has to see integers — an instance label map — and the vectors have to be
    built from what it returns. A target built first comes back geometrically consistent and
    physically wrong, and the loss falls either way, so nothing downstream can tell. The
    recording augmenter below is the only instrument that can.
    """
    import numpy as np
    from PIL import Image

    script = _flow_script()
    case = tmp_path / "case"
    (case / "images").mkdir(parents=True)
    (case / "masks").mkdir()
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(case / "images" / "case.png")
    for index, (row, column) in enumerate(((2, 2), (2, 9)), start=1):
        one = np.zeros((16, 16), np.uint8)
        one[row : row + 5, column : column + 5] = 255
        Image.fromarray(one).save(case / "masks" / f"{index}.png")

    seen = {}

    class Recording(script.Cases):
        def __init__(self):
            super().__init__([case], 16, augment=False, mode="diffusion")
            self.augmenter = self._record

        def _record(self, image, mask=None):
            seen["dtype"] = np.asarray(mask).dtype
            seen["ndim"] = np.asarray(mask).ndim
            seen["values"] = sorted(int(v) for v in np.unique(mask))
            return image, mask

    _, target = Recording()[0]

    assert np.issubdtype(seen["dtype"], np.integer), seen["dtype"]
    assert seen["ndim"] == 2, "a channel axis means this is not an instance label map"
    assert seen["values"] == [0, 1, 2], "the augmenter saw instances, not classes"
    assert tuple(target.shape) == (4, 16, 16), "and the four-channel target was built after"


def _discs(size=48, centres=((14, 14), (14, 34), (34, 24)), radius=7):
    """Three round objects, far enough apart that nothing about them is a near miss."""
    import numpy as np

    labels = np.zeros((size, size), dtype=np.int32)
    rows, columns = np.ogrid[:size, :size]
    for index, (row, column) in enumerate(centres, start=1):
        labels[(rows - row) ** 2 + (columns - column) ** 2 <= radius**2] = index
    return labels


def test_the_true_identity_oracle_does_not_shatter_a_mask_the_truth_does_not_cover():
    """The defect this function was written for, as a number.

    The oracle row started life as `flows_from_instances(labels)` handed to the model's own
    mask, and it scored **0.2220 against 0.5315 for the predicted field** with **18758
    objects back against 5999** in the truth. The cause is not the field's quality: a flow
    field is identically zero off its own objects, a zero field moves nothing, so every voxel
    the mask got wrong stands still and lands in a basin of its own. A mask one pixel too
    generous is enough to show it, and no model is needed - the same probe on twelve discs
    took 48 objects to 93, one spurious ring each.
    """
    import numpy as np
    from scipy import ndimage

    from pyplatypus.instances import flows_from_instances, instances_by_flows

    script = _flow_script()
    labels = _discs()
    mask = ndimage.binary_dilation(labels > 0, iterations=1)

    raw = instances_by_flows(flows_from_instances(labels, mode="diffusion"), mask)
    assert int(raw.max()) == 2 * int(labels.max()), (
        "the premise, and it is exact: the truth's own field gives every object a ring of "
        "stationary voxels, so three objects come back as six. If this stops being true the "
        "test below is no longer measuring anything"
    )

    fixed = instances_by_flows(script.truth_field_on(labels, mask, "diffusion"), mask)
    assert int(fixed.max()) == int(labels.max()), np.bincount(fixed.ravel())


def test_the_oracle_leaves_no_voxel_of_the_mask_without_a_vector_but_the_sinks():
    """What "replace exactly one part of the pipeline" means here, voxel by voxel.

    The field has to be defined wherever the decode will start from, and it is zero at a sink
    by construction - so the count of voxels without a vector is the number of objects and
    nothing else. One more than that is a hole the oracle would be reading as a result.
    """
    import numpy as np
    from scipy import ndimage

    script = _flow_script()
    labels = _discs()
    mask = ndimage.binary_dilation(labels > 0, iterations=2)

    field = script.truth_field_on(labels, mask, "diffusion")
    still = mask & (np.hypot(field[0], field[1]) == 0)
    assert int(still.sum()) == int(labels.max()), "one sink per object, no other hole"
    outside = ~(mask | (labels > 0))
    assert not (field[:, outside] != 0).any(), (
        "and nothing outside the support, which is `mask | truth` and not `mask` - said "
        "this way round because a dilated mask contains the truth, so a claim about `mask` "
        "alone would pass here by fixture rather than by code"
    )


def test_a_detached_piece_of_mask_joins_the_nearest_object_under_one_mode_only():
    """The oracle's one remaining blind spot, which the run counts rather than hides.

    Identity is assigned by distance, so a piece of predicted foreground touching no object
    at all still joins the nearest one. Whether it gets a *vector* depends on the target's
    shape and not on the oracle: a straight line to the sink needs no path through the
    object, while the diffusion is heat spread inside it and a disconnected piece receives
    none. The run prints how many such pieces there are beside the row, because each one
    lands as a basin the field did not earn.
    """
    import numpy as np

    script = _flow_script()
    labels = _discs()
    mask = labels > 0
    mask[44, 44] = True  # ten pixels clear of the nearest object

    straight = script.truth_field_on(labels, mask, "centroid")[:, 44, 44]
    assert np.hypot(*straight) == pytest.approx(1.0, abs=1e-5)
    assert straight[0] < 0 and straight[1] < 0, "pointing back at the object it was given to"

    heat = script.truth_field_on(labels, mask, "diffusion")[:, 44, 44]
    assert np.hypot(*heat) == 0.0, "and the counted blind spot is this, in this mode"


def test_the_oracle_s_support_is_the_mask_and_the_truth_so_a_cut_object_stays_one():
    """Why the support is a union rather than the mask, measured.

    A mask that cuts an object in half leaves one half without the sink, and the diffusion
    confined to that half would receive no heat - the same zero-field shatter the function
    exists to remove, arriving by a different door. Including the truth's own body in the
    support keeps the object connected for the heat to cross, and costs nothing, because
    `follow_flows` only ever starts from the mask.
    """
    import numpy as np

    script = _flow_script()
    from pyplatypus.instances import flows_from_instances, instances_by_flows

    labels = np.zeros((48, 48), dtype=np.int32)
    labels[10:38, 10:38] = 1
    mask = labels > 0
    mask[23:25] = False  # a band straight through it: two halves, one sink

    field = script.truth_field_on(labels, mask, "diffusion")
    assert int(instances_by_flows(field, mask).max()) == 1, "both halves reach the one sink"

    confined = flows_from_instances(np.where(mask, labels, 0), mode="diffusion")
    assert int(instances_by_flows(confined, mask).max()) == 2, (
        "the premise: with the support cut to the mask alone the far half has no heat and "
        "comes back as an object of its own"
    )


# ------------------------------------------------------------------ hv, #213's cheap test
#
# The two marker recipes live in `separation_ceiling.py` rather than in the package, because
# `#160` has not named `separation: hv` and the measurement below is the reason it may never
# need to. They are tested here for the same reason the flow script's functions are: nothing
# else decides what the maps mean.


def _ceiling_script():
    return __import__("separation_ceiling")


def _touching(width_one=6, width_two=14, height=12):
    """Two rectangles sharing an edge, of deliberately different sizes."""
    import numpy as np

    labels = np.zeros((height + 4, width_one + width_two + 4), dtype=np.int32)
    labels[2 : 2 + height, 2 : 2 + width_one] = 1
    labels[2 : 2 + height, 2 + width_one : 2 + width_one + width_two] = 2
    return labels


def test_the_hv_maps_run_from_minus_one_to_one_inside_every_object():
    """ "Normalised per object" is the whole of #213's target and it is one line to check.

    Both objects have to reach -1 and 1 along both axes however big they are, and the maps
    have to be exactly zero off the foreground - a non-zero background would put a gradient
    at every object's outer edge for the gradient recipe and a marker in the background for
    the centre one.
    """
    import numpy as np

    script = _ceiling_script()
    labels = _touching()
    maps = script.hv_from_instances(labels)

    assert maps.shape == (2, *labels.shape)
    assert maps.dtype == np.float32
    assert not maps[:, labels == 0].any(), "zero off the objects, both channels"
    for value in (1, 2):
        for channel in (0, 1):
            inside = maps[channel][labels == value]
            assert inside.min() == pytest.approx(-1.0, abs=1e-6), (value, channel)
            assert inside.max() == pytest.approx(1.0, abs=1e-6), (value, channel)


def test_the_first_hv_channel_varies_along_columns_and_the_second_along_rows():
    """The axis order, which is a contract and not a convention: both marker recipes take a
    derivative of channel 0 along columns and of channel 1 along rows, so a swap would give
    each of them the other's derivative and still look like a field."""
    import numpy as np

    script = _ceiling_script()
    labels = np.zeros((14, 14), dtype=np.int32)
    labels[3:11, 3:11] = 1
    maps = script.hv_from_instances(labels)

    assert np.ptp(maps[0][labels == 1].reshape(8, 8), axis=1).min() > 1.5, (
        "channel 0 sweeps the full range along every row of the object"
    )
    assert np.allclose(np.ptp(maps[0][labels == 1].reshape(8, 8), axis=0), 0.0), (
        "and does not change down a column, which is what makes it the horizontal map"
    )
    assert np.ptp(maps[1][labels == 1].reshape(8, 8), axis=0).min() > 1.5
    assert np.allclose(np.ptp(maps[1][labels == 1].reshape(8, 8), axis=1), 0.0)


def test_two_touching_objects_meet_at_a_jump_of_two_whatever_their_sizes():
    """Why the maps are normalised per object, as the number the normalisation buys.

    The gradient recipe calls a seam a seam because the horizontal map jumps from +1 to -1
    across it. Without the per-object normalisation that jump would be the sum of two object
    radii in pixels, so a small object beside a large one would present a smaller seam than
    the large one's own interior slope, and one global threshold could not serve both.
    """
    script = _ceiling_script()
    labels = _touching(width_one=6, width_two=14)
    maps = script.hv_from_instances(labels)

    seam = 2 + 6  # the first column of object two
    row = 2 + 6
    jump = maps[0][row, seam] - maps[0][row, seam - 1]
    assert jump == pytest.approx(-2.0, abs=0.35), (
        f"the seam is a jump of about two, measured {jump:.3f}"
    )


def test_the_centre_markers_are_one_per_object_and_inside_it():
    """Including the object whose centroid is not inside it, which is the case the straight-line
    flow mode is structurally wrong about and this recipe is not: the marker is where the maps
    are *near* zero, and those pixels belong to the object by construction."""
    import numpy as np
    from scipy import ndimage

    script = _ceiling_script()
    labels = np.zeros((40, 40), dtype=np.int32)
    rows, columns = np.ogrid[:40, :40]
    outer = (rows - 20) ** 2 + (columns - 20) ** 2 <= 14**2
    inner = (rows - 20) ** 2 + (columns - 14) ** 2 <= 10**2
    labels[outer & ~inner] = 1  # a crescent: its centroid falls in the bite
    assert labels[20, 20] == 0, "the premise: the centroid is outside the object"

    maps = script.hv_from_instances(labels)
    markers = script.markers_from_hv_centres(maps, labels > 0, 0.6)
    found = int(ndimage.label(markers, structure=np.ones((3, 3), bool))[1])

    assert markers.any(), "the object got a marker"
    assert not (markers & (labels == 0)).any(), "and every marker voxel is inside the object"
    assert found <= 2, f"a crescent may split into two arms, not into {found} markers"

    # And the condition is both maps at once rather than either. Either-or would give each
    # object a band across its whole width, the two bands would meet at the seam, and two
    # touching objects would come back as one marker - which is the one thing a marker
    # recipe for this problem must not do.
    pair = _touching()
    both = script.markers_from_hv_centres(script.hv_from_instances(pair), pair > 0, 0.6)
    assert int(ndimage.label(both, structure=np.ones((3, 3), bool))[1]) == 2


def test_the_gradient_markers_part_at_the_seam_the_maps_jump_across():
    """The premise of the published route, on the fixture its argument is about: the markers
    have to be two, and they have to be two because of the seam rather than because the
    rectangles were separate to begin with."""
    import numpy as np
    from scipy import ndimage

    script = _ceiling_script()
    labels = _touching()
    foreground = labels > 0
    assert ndimage.label(foreground, structure=np.ones((3, 3), bool))[1] == 1, (
        "the premise: the foreground is one connected blob, so connected components cannot "
        "tell these apart and anything that does is the hv signal"
    )

    markers, surface = script.markers_from_hv_gradient(
        script.hv_from_instances(labels), foreground, 0.6
    )
    found = int(ndimage.label(markers, structure=np.ones((3, 3), bool))[1])
    assert found == 2, f"two objects, {found} markers"
    assert surface.shape == labels.shape
    assert surface[foreground].max() > surface[~foreground].max(), (
        "the surface the watershed would climb is highest inside the foreground, where the "
        "seam is, and not in the background it was masked out of"
    )


# ------------------------------------------- the instance weights publishing script
#
# `publish_dsbowl_instance_weights.py` carries five refusals the other two publishing
# scripts do not need: the control that ties the split's names to the publishable
# download, the relocation that keeps the recorded run out of the way, the band that
# reads this run against that record, the one-mask check across decodings, and the
# sidecar's reserved names. Each is tested for being able to refuse - a control that
# cannot fail is what this project keeps finding - and the card is tested on numbers
# that are not the real ones, so a card printing anything it was not handed fails here.


def _publishing_script():
    import publish_dsbowl_instance_weights as script

    return script


def _decoded_rows(**moved) -> list[dict]:
    """Three decodings of one trained model, at the recorded numbers unless moved."""
    script = _publishing_script()
    return [
        {
            "model": "rim",
            "separation": "rim",
            "decode": name,
            "foreground_dice": 0.9159,
            "dice": 0.8550,
            "average_precision": moved.get(name, recorded),
            "predicted_objects": 5414,
            "truth_objects": script.TRUTH_OBJECTS,
        }
        for name, recorded in script.RECORDED.items()
    ]


def test_the_band_accepts_a_reproduction_and_refuses_a_changed_run():
    """Two-sided, because the direction that is never checked is the flattering one.

    A run below the band is broken rather than worse - run B's pretrained encoder landed
    0.1053 away - and one above it is a different recipe than the card would describe. The
    numbers here are the two ends the bound was calibrated between: one spread of 0.009 is
    inside it and run B's measured 0.4389 is not.
    """
    script = _publishing_script()
    script.check_band(_decoded_rows(), script.SPREAD, 3.0)
    script.check_band(_decoded_rows(nearest_seed=0.5442 + script.SPREAD), script.SPREAD, 3.0)

    with pytest.raises(SystemExit, match="nearest_seed: 0.4389"):
        script.check_band(_decoded_rows(nearest_seed=0.4389), script.SPREAD, 3.0)
    with pytest.raises(SystemExit, match=r"\+0.0558"):
        script.check_band(_decoded_rows(nearest_seed=0.6000), script.SPREAD, 3.0)


def test_the_split_is_checked_by_its_object_count_and_not_by_its_size():
    """The case count is the weak half: any 134 directories give it.

    5999 ground-truth objects over these 134 cases is the fingerprint, and it is checked one
    object away rather than at some margin - a different set of images that happened to hold
    5998 nuclei is exactly the failure `dsbowl-unet`'s card already shipped once, where two
    different 134s differed by 0.01 of Dice.
    """
    script = _publishing_script()
    script.check_split(_decoded_rows())

    one_short = [{**row, "truth_objects": script.TRUTH_OBJECTS - 1} for row in _decoded_rows()]
    with pytest.raises(SystemExit, match="look comparable and are not"):
        script.check_split(one_short)


def test_three_decodings_of_one_model_report_one_mask():
    """The card's central claim is three numbers from one file, and this is its control.

    Two different foreground Dice values in these rows would mean two trained models, and
    then the card's table is three scores of three things rather than three readings of one.
    """
    script = _publishing_script()
    script.check_one_mask(_decoded_rows())

    rows = _decoded_rows()
    rows[1] = {**rows[1], "foreground_dice": 0.9097}
    with pytest.raises(SystemExit, match="not three decodings of one trained model"):
        script.check_one_mask(rows)


def _nested_copy(root: Path, cases: list[str], image: bytes = b"image") -> None:
    """Cases in the layout both copies use, as bytes rather than pictures.

    The control hashes files and never decodes them, so bytes are the honest fixture: a PNG
    here would test PIL's writer as well, and the thing under test is whether two copies of
    one dataset are the same data.
    """
    for name in cases:
        (root / name / "images").mkdir(parents=True)
        (root / name / "masks").mkdir(parents=True)
        (root / name / "images" / f"{name}.png").write_bytes(image)
        for index, blob in enumerate((b"first object", b"second object")):
            (root / name / "masks" / f"{index}.png").write_bytes(blob)


def test_the_data_control_refuses_a_copy_that_is_not_the_same_data(tmp_path):
    """The licence chain and the split's names meet here, so this is the join that must hold.

    The split is named by the directories of a 2022 Kaggle download and the weights are
    trained on BBBC038v1; if the two copies differ, the card's provenance describes one and
    its numbers come from the other. Measured on the real pair in the run: 670 cases
    byte-identical in both the images and the per-object masks.
    """
    import shutil

    script = _publishing_script()
    local, broad = tmp_path / "local", tmp_path / "broad"
    _nested_copy(local / "stage1_train", ["alpha"])
    _nested_copy(local / "stage1_validation", ["beta"])
    _nested_copy(broad, ["alpha", "beta"])
    names = {"train": ["alpha"], "validation": ["beta"]}

    script.same_data(names, broad, local, ("images", "masks"))

    (broad / "beta" / "images" / "beta.png").write_bytes(b"a different picture")
    with pytest.raises(SystemExit, match="the image differs"):
        script.same_data(names, broad, local, ("images", "masks"))

    # The image is restored and a mask is changed instead: the masks are what the model
    # learns from, so a control that only compared images would answer the adjacent
    # question and answer it reassuringly.
    (broad / "beta" / "images" / "beta.png").write_bytes(b"image")
    (broad / "beta" / "masks" / "1.png").write_bytes(b"a third object")
    with pytest.raises(SystemExit, match="learns from these files"):
        script.same_data(names, broad, local, ("images", "masks"))

    (broad / "beta" / "masks" / "1.png").write_bytes(b"second object")
    script.same_data(names, broad, local, ("images", "masks"))

    shutil.rmtree(broad / "beta")
    with pytest.raises(SystemExit, match="cannot be taken by name"):
        script.same_data(names, broad, local, ("images", "masks"))


def test_the_split_refuses_a_case_in_both_halves(tmp_path):
    """A leak does not show up in a score, which is why it is refused rather than reported."""
    script = _publishing_script()
    _nested_copy(tmp_path / "stage1_train", ["alpha", "shared"])
    _nested_copy(tmp_path / "stage1_validation", ["shared"])

    with pytest.raises(SystemExit, match="in both halves"):
        script.split_names(tmp_path)


def test_the_split_refuses_a_validation_half_of_the_wrong_size(tmp_path):
    """Every number this card is read against was measured on 134 cases."""
    script = _publishing_script()
    _nested_copy(tmp_path / "stage1_train", ["alpha"])
    _nested_copy(tmp_path / "stage1_validation", ["beta"])

    with pytest.raises(SystemExit, match="was measured on 134"):
        script.split_names(tmp_path)


def test_relocating_keeps_the_recorded_run_out_of_the_way_and_says_when_it_cannot(tmp_path):
    """`#215`'s record is the band's centre, and the file being published writes over it.

    `examples/data_science_bowl_instances.yaml` checkpoints into `examples/output/rim/`,
    where the recorded run and its own checkpoint live. Running it unchanged would overwrite
    the control with the thing being controlled, and the only symptom would be that every
    later run agreed with itself. The second half is the refusal: a second callback writing
    there is not rewritten, and the check is what notices rather than its absence.
    """
    import yaml

    script = _publishing_script()
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parent.parent / "examples" / "data_science_bowl_instances.yaml"
        ).read_text()
    )
    work = tmp_path / "work"
    moved = script.relocate(config, work)
    written = json.dumps(moved)
    assert "examples/output/rim" not in written and "examples/output/flows" not in written

    # The second half reads the structure rather than the serialised text, and the first half
    # is right to read the text. `json.dumps` escapes a Windows path's separators, so
    # `str(work)` is not a substring of the blob however correct every path in it is - this
    # test failed on both Windows jobs and nowhere else for exactly that reason. The negative
    # half is portable because those two strings come out of the YAML verbatim: they can only
    # still be in there if nothing rewrote them, which is the question it is asking.
    assert Path(moved["output_dir"]) == work
    checkpoints = [
        callback["path"]
        for model in moved["models"]
        for callback in model.get("callbacks", [])
        if callback.get("name") == "model_checkpoint"
    ]
    assert len(checkpoints) == len(config["models"]), "every model's checkpoint is relocated"
    assert all(Path(path).parent == work for path in checkpoints)

    config["models"][0]["callbacks"].append(
        {"name": "csv_logger", "path": "examples/output/rim/history.csv"}
    )
    with pytest.raises(SystemExit, match="still writes into examples/output/rim"):
        script.relocate(config, tmp_path / "work")


def test_the_sidecar_cannot_shadow_what_the_weights_say_they_are_for():
    """A sidecar key colliding with the fingerprint is a weights file that cannot load.

    `export_weights` merges the extras over the specification's own fingerprint and
    `_refuse_mismatch` reads those same names back, so an extra called `separation` - the
    obvious name for the block's settings - would replace the string `rim` with a dict and
    every later load would refuse the file for disagreeing with itself. The reserved names
    come from the specification, so this also asserts that `separation` is one of them.
    """
    import yaml

    script = _publishing_script()
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parent.parent / "examples" / "data_science_bowl_instances.yaml"
        ).read_text()
    )
    spec = from_dict(config, check_paths=False)
    assert "separation" in spec.models[0].weights_fingerprint()

    script.refuse_a_shadowed_fingerprint(
        {"data_licence": "CC0 1.0", "separation_settings": {"name": "rim"}}, spec
    )
    for reserved in ("separation", "filters", "blocks", "input_shape", "task", "n_class"):
        with pytest.raises(SystemExit, match="refuse to load"):
            script.refuse_a_shadowed_fingerprint({reserved: "anything"}, spec)


def _measured(rows: list[dict]) -> dict:
    """Everything the card reads, with numbers that are not this model's."""
    return {
        "pyplatypus": "9.9.9",
        "seed": 3,
        "epochs": 7,
        "minutes": 11.0,
        "data_source": "https://example.invalid/stage1_train.zip",
        "citation": "Somebody et al., a paper, 1999.",
        "split": {"train": 11, "validation": 22},
        "separation_settings": {"name": "rim", "width": 2},
        "decodings": [row["decode"] for row in rows],
        "table": rows,
    }


def test_the_card_and_the_registry_entry_carry_the_numbers_they_were_handed():
    """Both are generated from the run, so neither may contain a number nobody measured.

    The scores here are deliberately not this model's: a card or a registry description
    printing 0.5442 would be printing a number from this test's own absence, which is how a
    model card comes to describe a run that never happened.
    """
    script = _publishing_script()
    rows = _decoded_rows(nearest_seed=0.5111, watershed=0.5022, connected_components=0.4033)
    measured = _measured(rows)

    card = script.card(measured)
    for name in ("nearest_seed", "watershed", "connected_components"):
        assert name in card, "a number without its decoding is not a property of the weights"
    for score in ("0.5111", "0.5022", "0.4033"):
        assert score in card
    assert "+0.1078" in card, "the signal is the best row over the yardstick, computed here"
    assert "585 objects of 5999 are not found" in " ".join(card.split())
    assert "pyplatypus 9.9.9" in card and "seed 3" in card
    assert "0.5442" not in card, "nothing on the card may come from outside the run"

    entry = script.registry_entry(measured)
    for score in ("0.5111", "0.5022", "0.4033"):
        assert score in entry
    assert "dsbowl-instances.safetensors" in entry
    assert "<the commit" in entry, "the revision is the author's to fill in after the upload"


# --- HC18: the example whose answer is a length, so its arithmetic is the measurement -----
#
# The three measurement scripts before this one carry no test but the import above, and that
# was defensible: each ends in a Dice the engine computed. This one ends in millimetres it
# computes itself - an ellipse fitted from a region's moments, Ramanujan's circumference, a
# pixel size out of a CSV - so the number nobody else checks is the number the example is
# for. These hold that arithmetic to something it cannot be right about by accident.


def _hc18():
    import measure_fetal_head_circumference as script

    return script


def test_the_head_circumference_configuration_is_accepted():
    """And it divides by examination rather than by row, which is the whole of the split.

    Three assertions, because each guards a different mistake the configuration could make
    silently: a two-way split would report the millimetres on the data early stopping
    watched, a `group_by` of `None` would put a near-duplicate on both sides of the
    division, and a pattern that matches no key raises inside the engine rather than here.
    """
    script = _hc18()
    arguments = _declared_arguments("measure_fetal_head_circumference")
    config = script.specification(Path("cases.csv"), arguments)
    _validate(config)

    split = config["data"]["split"]
    assert len(split["fractions"]) == 3, (
        "the millimetres are reported on a third split, so they do not come off the data "
        f"`early_stopping` watched; got {split['fractions']}"
    )
    assert split["group_by"] == script.GROUP_BY
    import re as _re

    for key in ("000_HC", "010_2HC", "805_HC"):
        found = _re.match(split["group_by"], key)
        assert found is not None, f"{key} would make the engine refuse the whole split"
    assert _re.match(split["group_by"], "010_2HC").group(1) == "010", (
        "the group is the examination, so a repeat photograph of one examination must not "
        "become its own group"
    )


def test_an_open_curve_fills_to_itself_which_is_what_the_guard_reads():
    """The four excluded annotations, as the shape the exclusion is triggered by.

    `fill` cannot close a curve that leaves the frame, and the failure is silent: the mask
    comes back as the one-pixel line. So the guard compares the filled area against the
    drawn area, and this holds both directions of that comparison - a closed curve grows by
    a large factor, an open one does not grow at all.
    """
    script = _hc18()
    inside = filled_ellipse((400, 400), (200, 200), 150.0, 110.0)
    from scipy.ndimage import binary_erosion

    closed = inside ^ binary_erosion(inside)
    assert script.fill(closed).sum() > 10.0 * closed.sum()

    # The real shape of it: the ellipse runs off the top of the image, so the curve is open
    # where the frame cuts it.
    cut = filled_ellipse((400, 400), (60, 200), 150.0, 110.0)
    open_curve = cut ^ binary_erosion(cut)
    open_curve[0, :] = False
    assert script.fill(open_curve).sum() < 2.0 * open_curve.sum(), (
        "an open curve filled to a region; the guard in `prepare` would then write a mask "
        "that is a line and train a model to find one"
    )


def test_the_published_hc18_weights_are_the_run_the_example_documents():
    """Every argument that shapes the model has the same default in both scripts.

    `publish_hc18_weights.py` imports `specification()` rather than copying it, so the
    *structure* cannot drift - but it declares its own flags, and a `--filters 32` here
    would publish a model the example does not describe while both files still looked right.
    The flags are read out of the source, so an argument added to either script is covered
    by arriving rather than by being listed here.
    """
    example = _declared_arguments("measure_fetal_head_circumference")
    publish = _declared_arguments("publish_hc18_weights")

    shapes_the_model = (
        "height",
        "width",
        "epochs",
        "patience",
        "batch",
        "filters",
        "rate",
        "loss",
        "seed",
    )
    for name in shapes_the_model:
        assert hasattr(example, name), f"the example no longer declares --{name}"
        assert getattr(publish, name) == getattr(example, name), (
            f"--{name} is {getattr(publish, name)!r} in the publishing script and "
            f"{getattr(example, name)!r} in the example, so the published weights would not "
            "be the run the example documents"
        )

    # And the control: the two scripts do differ, or this test is comparing a file with
    # itself. `--out` is the difference that matters - the example writes measurements, this
    # writes weights.
    assert publish.out != example.out


def test_the_hc18_checkpoint_watches_what_the_stopping_rule_watches():
    """Or the file on disk is from one epoch and the published weights from another.

    `restore_best` hands back the epoch `early_stopping` judged best, so a checkpoint
    monitoring anything else leaves two things named for the same run and measured on
    different ones. The publishing script reads the monitor off the stopping rule rather
    than repeating it, and this asserts that it did.
    """
    import publish_hc18_weights as script

    example = _declared_arguments("measure_fetal_head_circumference")
    base = script.example_module().specification(Path("cases.csv"), example)
    configured = script.with_checkpoint(base, Path("/tmp/does-not-matter"))

    callbacks = {entry["name"]: entry for entry in configured["models"][0]["callbacks"]}
    assert "model_checkpoint" in callbacks, "the checkpoint was not added"
    assert callbacks["model_checkpoint"]["monitor"] == callbacks["early_stopping"]["monitor"]
    assert callbacks["model_checkpoint"]["save_best_only"] is True

    # The whole configuration still has to be one the engine accepts, with the callback on
    # it: a checkpoint is the kind of addition that validates on its own and fails in a pair.
    _validate(configured)


def _hc18_measured() -> dict:
    """Everything the HC18 card reads, with numbers that are not this model's.

    Deliberately not the published figures, for the reason `_measured` above gives: a card
    tested on the numbers it really prints passes when it prints a constant, and the failure
    this guards is a card describing a run that never happened.
    """
    return {
        "summary": {
            "n": 111,
            "dice": 0.8111,
            "dice_worst": 0.2111,
            "mae_mm": 9.111,
            "bias_mm": 3.111,
            "model_mae_mm": 8.111,
            "model_bias_mm": 1.111,
            "worst_mm": 71.111,
        },
        "spread": {"median_mm": 6.111, "p75_mm": 11.111, "p90_mm": 21.111},
        "pipeline_control": {
            "n": 112,
            "line_width_px": 3.111,
            "predicted_bias_mm": 1.311,
            "bias_mm": 1.411,
            "correlation": 0.9111,
            "residual_mae_mm": 0.111,
        },
        "split": {"train": 311, "validation": 111, "test": 112},
        "disagreement": {
            "spearman": 0.711,
            "best_decile_by_dice": {"n": 11, "dice": 0.9111, "mae_mm": 5.111, "worst_mm": 9.111},
        },
        "published_but_wrong": {
            "dice_at_least": 0.95,
            "matters_mm": 2.0,
            "n_publishable": 71,
            "n_over": 31,
            "share_over": 31 / 71,
            "worst_case": "111_HC",
            "worst_dice": 0.9611,
            "worst_error_mm": 7.111,
        },
        "epochs": 11,
        "minutes": 11.1,
        "trained_with": "pyplatypus 9.9.9",
    }


def test_the_card_says_the_spearman_is_one_draw_and_says_what_the_others_were():
    """The one figure on this card measured on a single run and quoted as a property.

    Three runs of the one recipe gave 0.62, 0.435 and 0.153, and this script's own docstring
    said "about 0.6" for a day while the run it publishes measured 0.15 - so the card quotes
    the range and the table, and the rows are generated from `RUNS` rather than typed. Every
    row has to reach the card, or editing the constant and leaving the prose behind is
    possible again.
    """
    import publish_hc18_weights as script

    card = script.card(_hc18_measured())
    flat = " ".join(card.split())

    for label, dice, rho, share in script.RUNS:
        assert label in flat, f"the row for {label!r} is not on the card"
        assert f"{dice:.4f}" in flat, f"{label!r} lost its Dice"
        assert f"{rho:.2f}" in flat, f"{label!r} lost its Spearman"
        assert ("not counted" if share is None else f"{share * 100:.0f}%") in flat

    assert f"moved from {max(r[2] for r in script.RUNS):.2f}" in flat
    assert f"to {min(r[2] for r in script.RUNS):.2f}" in flat

    # And the control, which is the whole reason the fixture's numbers are invented: the
    # published run's own Spearman is the fixture's and not one of the recorded three, so a
    # card that printed `RUNS`'s last row where it means *this* run would fail here.
    assert "0.71" in flat, "the run's own Spearman is not on the card"
    assert "0.153" not in flat, "a recorded run's figure is standing in for this run's"


def test_the_card_is_not_indented_because_an_indented_card_is_a_code_block():
    """`textwrap.dedent` strips the COMMON prefix, and a block interpolated at column 0
    leaves none - so adding the spread note indented the whole card by eight spaces and the
    Hub would have rendered the entire section as preformatted text. Found by reading the
    generated card, which is the only thing that could have found it.
    """
    import publish_hc18_weights as script

    card = script.card(_hc18_measured())
    lines = card.splitlines()

    assert lines[0] == "## hc18-unet", f"the card does not start at column 0: {lines[0]!r}"
    assert not any(line.startswith("    ") and "=" not in line for line in lines), (
        "a line of the card is indented four spaces or more outside the code block, which "
        "markdown reads as preformatted text"
    )


def test_the_hc18_registry_entry_carries_the_numbers_it_was_handed():
    """`UPLOAD.md` said this script printed the registry entry for a day before it did.

    The entry is generated for the same reason the card is: the description is the only part
    of a registry entry carrying numbers, and one copied out of a terminal is one nothing
    holds to the run. The fixture's numbers are deliberately not this model's, so an entry
    printing a constant fails here rather than passing on the real figures.
    """
    import publish_hc18_weights as script

    measured = _hc18_measured()
    entry = script.registry_entry(measured, (123, 456))
    flat = " ".join(entry.split())

    assert '"hc18-unet": Published(' in entry
    assert 'filename="hc18-unet.safetensors"' in entry
    assert "ls-remote" in entry, "the revision is a placeholder until the upload has a commit"

    assert "123x456" in flat, "the input shape is the one it was handed"
    for number in (
        f"{measured['spread']['median_mm']:.2f} mm",
        f"{measured['summary']['mae_mm']:.2f} mm",
        f"{measured['summary']['worst_mm']:.2f} mm",
        f"{measured['summary']['model_mae_mm']:.2f} mm",
        f"{measured['summary']['dice']:.4f}",
    ):
        assert number in flat, f"{number} is not in the entry"
    assert (
        f"{measured['published_but_wrong']['n_over']} of the "
        f"{measured['published_but_wrong']['n_publishable']} cases"
    ) in flat

    # The two claims the entry exists to make, in the order that is the claim.
    assert flat.index("millimetres") < flat.index("Dice")
    assert "Do not choose these weights on the Dice" in flat
    assert "not MIT" in flat

    # And the control: none of the published run's own figures may appear, or the entry is
    # printing something it was not handed.
    for published in ("2.44 mm", "3.84 mm", "0.9494", "85 of the 144"):
        assert published not in flat, f"{published} came from outside this run"
